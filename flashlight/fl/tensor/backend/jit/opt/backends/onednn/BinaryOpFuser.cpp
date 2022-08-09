/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <iostream>

#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/CustomNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/IndexNode.h"
#include "flashlight/fl/tensor/backend/jit/opt/backends/onednn/BinaryOpFuser.h"
#include "flashlight/fl/tensor/backend/onednn/OneDnnBackend.h"
#include "flashlight/fl/tensor/backend/onednn/OneDnnTensor.h"
#include "flashlight/fl/tensor/backend/onednn/Utils.h"

#include "dnnl.hpp"

namespace fl {

struct BinaryOpFuser::BinopInfo {
  std::shared_ptr<Node> otherInputNode;
  BinaryOp op;
};

namespace {

// https://github.com/oneapi-src/oneDNN/blob/adeda9fcc20149effb1bffc051262810e9f3138c/include/oneapi/dnnl/dnnl_types.h#L2914
static constexpr unsigned kOneDnnMaxNumPostOps = 32;

dnnl::memory::data_type getOneDnnTypeWithLargestRange(
    const std::vector<Tensor>& tensors) {
  assert(!tensors.empty());
  dnnl::memory::data_type largestType =
    detail::flToOneDnnType(tensors.front().type());
  for (unsigned i = 1; i < tensors.size(); i++) {
    auto otherType = detail::flToOneDnnType(tensors[i].type());
    largestType = detail::getTypeWithLargerRange(largestType, otherType);
  }
  return largestType;
}

bool allHaveSameShapes(const std::vector<Tensor>& tensors) {
  if (tensors.empty()) {
    return true;
  }
  const auto& shape = tensors.front().shape();
  for (unsigned i = 1; i < tensors.size(); i++) {
    if (shape != tensors[i].shape()) {
      return false;
    }
  }
  return true;
}

dnnl::algorithm binopToOneDnnAlg(const BinaryOp op) {
  switch(op) {
    case BinaryOp::Add: return dnnl::algorithm::binary_add;
    case BinaryOp::Sub: return dnnl::algorithm::binary_sub;
    case BinaryOp::Mul: return dnnl::algorithm::binary_mul;
    case BinaryOp::Div: return dnnl::algorithm::binary_div;
    case BinaryOp::Eq: return dnnl::algorithm::binary_eq;
    case BinaryOp::Neq: return dnnl::algorithm::binary_ne;
    case BinaryOp::Gt: return dnnl::algorithm::binary_gt;
    case BinaryOp::Gte: return dnnl::algorithm::binary_ge;
    case BinaryOp::Lt: return dnnl::algorithm::binary_lt;
    case BinaryOp::Lte: return dnnl::algorithm::binary_le;
  }
  throw std::runtime_error("Unsupported binary operation type");
}

bool isOpCommutative(const BinaryOp op) {
  switch(op) {
    case BinaryOp::Add:
    case BinaryOp::Mul:
    case BinaryOp::Eq:
    case BinaryOp::Neq:
      return true;
    case BinaryOp::Sub:
    case BinaryOp::Div:
    case BinaryOp::Gt:
    case BinaryOp::Gte:
    case BinaryOp::Lt:
    case BinaryOp::Lte:
      return false;
  }
  throw std::runtime_error("Unsupported binary operation type");
}

bool isOpFusable(const BinaryOp op) {
  switch(op) {
    case BinaryOp::Add:
    case BinaryOp::Mul:
    case BinaryOp::Sub:
    case BinaryOp::Div:
    case BinaryOp::Eq:
    case BinaryOp::Neq:
    case BinaryOp::Gt:
    case BinaryOp::Gte:
    case BinaryOp::Lt:
    case BinaryOp::Lte:
      return true;
  }
  throw std::runtime_error("Unsupported binary operation type");
}

bool isNodeFusable(const BinaryNode& node) {
  // TODO consider whether casting fits into this, do we need type inference?
  return isOpFusable(node.op());
}

bool isFusionProfitable(const std::shared_ptr<Node> node) {
  // no cost to fuse non-binary node in this optimziation since they'll be used
  // as inputs in the fused node, and thus we don't skip their materialization.
  return !node->isBinary() || node->getUseCount() == 1;
  // TODO Even if we have > 1 use, it might be possible & profitable to fuse
  // Need a better correctness & cost model.
}

// forward decl for mutually recursive functions
unsigned propogateProfitableFuseChainLengths(
    std::shared_ptr<Node> node,
    std::unordered_map<Node*, unsigned>& lengthMap);

// NOTE caller is responsible for updating `node` in `lengthMap`
unsigned propogateProfitableFuseChainLengths(
    const BinaryNode& node,
    std::unordered_map<Node*, unsigned>& lengthMap) {
  auto lhsLength = propogateProfitableFuseChainLengths(node.lhs(), lengthMap);
  auto rhsLength = propogateProfitableFuseChainLengths(node.rhs(), lengthMap);
  auto curHeight = 0u;
  // try fuse onto lhs or rhs
  if (isNodeFusable(node) && isFusionProfitable(node.lhs())) {
    curHeight = 1 + lhsLength;
    if (isOpCommutative(node.op()) && isFusionProfitable(node.rhs())) {
      curHeight = std::max(curHeight, rhsLength);
    }
  }
  return curHeight;
}

unsigned propogateProfitableFuseChainLengths(
    std::shared_ptr<Node> node,
    std::unordered_map<Node*, unsigned>& lengthMap) {
  // skip if already visited
  const auto& iter = lengthMap.find(node.get());
  if (iter != lengthMap.end()) {
    return iter->second;
  }
  auto curHeight = 0;
  switch (node->type()) {
    case NodeType::Binary: {
      curHeight = propogateProfitableFuseChainLengths(
          node->impl<BinaryNode>(), lengthMap);
      break;
    }
    case NodeType::Custom:
    // TODO go crazy and optimize the tensor indices
    case NodeType::Index:
    case NodeType::IndexedMerge:
    case NodeType::Scalar:
    case NodeType::Value: {
       // chain stops at these nodes (they are leaf data)
       for (const auto& inputNode : node->inputs()) {
         propogateProfitableFuseChainLengths(inputNode, lengthMap);
       }
       break;
     }
    default:
        throw std::runtime_error("Unknown node type");
  }
  lengthMap[node.get()] = curHeight;
  return curHeight;
}

std::unordered_map<Node*, unsigned> getFusableChainLengths(
    std::shared_ptr<Node> root) {
  std::unordered_map<Node*, unsigned> lengthMap;
  propogateProfitableFuseChainLengths(root, lengthMap);
  return lengthMap;
}

} // namespace

BinaryOpFuser::BinaryOpFuser(std::shared_ptr<Node> root)
 : profitableFuseChainLengths_(getFusableChainLengths(root)) {}

std::shared_ptr<Node> BinaryOpFuser::rewriteFrom(
    std::shared_ptr<Node> node) {
  std::vector<BinopInfo> accumualtedBinops;
  return rewriteFrom(node, accumualtedBinops);
}

// In the following case `node` is `x1`, assume we started from op2.
//
// x0  x1
//  \  /
//   op1  x2
//     \  /
//     op2
//
// accumualtedBinops: { { op1, x1 }, {op2, x2} }
//
// Accumulate as many binop as possible in `accumualtedBinops` and fuse them with
// `node` at the end. Then _keep_ applying `rewriteFrom` to children.
std::shared_ptr<Node> BinaryOpFuser::rewriteFrom(
    std::shared_ptr<Node> node,
    std::vector<BinopInfo>& accumualtedBinops) {
  if (visited_.find(node.get()) != visited_.end()) {
    return fuseAccumulatedBinops(node, accumualtedBinops);
  }
  visited_.insert(node.get());

  // can't fuse anything down the input chain with this one
  if (profitableFuseChainLengths_.at(node.get()) == 0) {
    // TODO 1st pass can record more information to help us here.
    // Consider having it return `Action` enum instead of chain length, then we
    // can centralize "how to fuse" into an Analyzer oracle, and have an
    // Rewriter that simply look up and execute those actions.
    node =
      node->mapInputs([this](auto input) { return rewriteFrom(input); });

  } else if (node->isBinary() && accumualtedBinops.size() <= kOneDnnMaxNumPostOps) {
    const auto& binaryNode = node->impl<BinaryNode>();
    const auto lhs = binaryNode.lhs();
    const auto rhs = binaryNode.rhs();
    const auto op = binaryNode.op();
    const auto lhsChainLength = profitableFuseChainLengths_.at(lhs.get());
    const auto rhsChainLength = profitableFuseChainLengths_.at(rhs.get());
    // At a binary node we fuse the longer chain into this binary node; since
    // we passed the initial length check, we must be able to fuse along rhs/lhs
    if (lhsChainLength >= rhsChainLength) {
      // optimize the other input before fusing it
      accumualtedBinops.push_back({ rewriteFrom(rhs), op });
      return rewriteFrom(lhs, accumualtedBinops);
    } else {
      // optimize the other input before fusing it
      accumualtedBinops.push_back({ rewriteFrom(lhs), op });
      return rewriteFrom(rhs, accumualtedBinops);
    }

  } else if (accumualtedBinops.size() > kOneDnnMaxNumPostOps) {
    // Must fuse accumulated binops -- 1 as regular binop, the rest as post-ops.
    // However, we should still try optimizing the lhs input.

    // sigh, we need to restart the process with empty accumulation
    visited_.erase(node.get());
    // NOTE This will not loop-forever because `accumualtedBinops` starts as empty.
    node = rewriteFrom(node);
  }
  return fuseAccumulatedBinops(node, accumualtedBinops);
}

std::shared_ptr<Node> BinaryOpFuser::fuseAccumulatedBinops(
    std::shared_ptr<Node> node,
    std::vector<BinopInfo>& accumualtedBinops) {
  if (accumualtedBinops.empty()) {
    return node;
  }
  // NOTE technically fusing 1 binop is just recreating the original binop, but
  // we let it be to keep algorithmic simplicity.

  // In the following case `node` is `x1`
  //
  // x1  x2
  //  \  /
  //   op1  x3
  //     \  /
  //     op2
  // becomes
  // inputNodes: { x1, x2, x3 }
  // algs:       { op1, op2 }
  std::vector<std::shared_ptr<Node>> inputNodes{node};
  std::vector<dnnl::algorithm> algs;
  for (int i = accumualtedBinops.size() - 1; i >= 0; i--) {
    const auto& info = accumualtedBinops[i];
    algs.push_back(binopToOneDnnAlg(info.op));
    inputNodes.push_back(info.otherInputNode);
  }

  // TODO refactor with common logic in OneDnnBackend
  auto evalFunc = [algs = std::move(algs)](const std::vector<Tensor>& inputs) {
    assert(inputs.size() == algs.size() + 1);

    // TODO shallow copy
    const Tensor* lhs = &inputs[0];
    const Tensor* rhs = &inputs[1];
    // TODO
    // - support same-ndim broadcast (if we force OneDNN Tensor to always have
    //   same # of internal dims, like ArrayFire, we can easily support general
    //   broadcast)
    if (!allHaveSameShapes(inputs)) {
      throw std::runtime_error("[BinaryOpFuser] inputs must have same shapes");
    }
    const auto& dstShape = lhs->shape();
    const auto dstType = getOneDnnTypeWithLargestRange(inputs);

    auto& backend = OneDnnBackend::getInstance();
    auto& engine = backend.engine();

    // prepare memories
    dnnl::algorithm alg = algs.front();
    auto& lhsMem = toOneDnnTensor(*lhs).memory();
    auto& rhsMem = toOneDnnTensor(*rhs).memory();
    const auto lhsMemDesc = lhsMem.get_desc();
    const auto rhsMemDesc = rhsMem.get_desc();
    // TODO reuse `lhsMem` or `rhsMem` if they don't need to be materialized;
    // need to invalidate the tensor result on the node though.
    const auto dstMemDesc =
      detail::oneDnnContiguousMemDescFromShape(dstShape, dstType);
    auto dstMem = dnnl::memory(dstMemDesc, engine);

    // prepare part of primitive
    const dnnl::binary::desc binaryDesc(
        alg, lhsMemDesc, rhsMemDesc, dstMemDesc);

    // prepare part of arguments
    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC_0, lhsMem},
        {DNNL_ARG_SRC_1, rhsMem},
        {DNNL_ARG_DST, dstMem},
    };

    // prepare post ops
    dnnl::post_ops binops;
    for (unsigned i = 1; i < algs.size(); i++) {
      // set up the other input for post-op
      auto& otherMem = toOneDnnTensor(inputs[i+1]).memory();
      binops.append_binary(algs[i], otherMem.get_desc());
      args.insert( // DNNL_ARG_SRC_1 feels totally arbitrary...
          {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i-1) | DNNL_ARG_SRC_1, otherMem});
    }

    // finish building primitive
    dnnl::primitive_attr binaryAttr;
    binaryAttr.set_post_ops(binops);
    const auto binaryPrimtiveDesc =
        dnnl::binary::primitive_desc(binaryDesc, binaryAttr, engine);
    const auto binaryPrimitive = dnnl::binary(binaryPrimtiveDesc);

    // execute primitive
    binaryPrimitive.execute(backend.nativeStream(), args);
    return toTensor<OneDnnTensor>(dstShape, std::move(dstMem));
  };

  return CustomNode::create(
      "OneDnnFusedBinaryOp",
      std::move(inputNodes),
      Shape(node->shape()),
      std::move(evalFunc));
}

std::shared_ptr<Node> BinaryOpFuser::apply(std::shared_ptr<Node> root) {
  BinaryOpFuser fuser(root);
  return fuser.rewriteFrom(root);
}

} // namespace fl
