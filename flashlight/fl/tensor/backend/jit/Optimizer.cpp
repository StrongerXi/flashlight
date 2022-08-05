/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/Optimizer.h"

#include "flashlight/fl/tensor/backend/jit/Utils.h"
#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/CustomNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"

namespace fl {

namespace {

template<typename T>
T mergeScalars(T lhs, T rhs, BinaryOp op) {
  switch (op) {
    case BinaryOp::Add: return lhs + rhs;
    case BinaryOp::Sub: return lhs - rhs;
    case BinaryOp::Mul: return lhs * rhs;
    case BinaryOp::Div: return lhs / rhs;
  }
  throw std::runtime_error("Unknown binary operation type");
}

template<typename T>
std::shared_ptr<ScalarNode> mergeScalarNodes(
    const ScalarNode& lhs,
    const ScalarNode& rhs,
    BinaryOp op,
    const Shape& shape,
    dtype type) {
  T lhsVal = lhs.scalar<T>();
  T rhsVal = rhs.scalar<T>();
  T resVal = mergeScalars(lhsVal, rhsVal, op);
  return ScalarNode::create(shape, type, resVal);
}

std::shared_ptr<ScalarNode> mergeScalarNodes(
    const ScalarNode& lhs,
    const ScalarNode& rhs,
    BinaryOp op,
    const Shape& shape,
    dtype type) {
  switch (type) {
    case dtype::f16:
      throw std::runtime_error("[mergeScalarNodes] f16 is unsupported");
    case dtype::f32: return mergeScalarNodes<float>(lhs, rhs, op, shape, type);
    case dtype::f64: return mergeScalarNodes<double>(lhs, rhs, op, shape, type);
    case dtype::b8: return mergeScalarNodes<char>(lhs, rhs, op, shape, type);
    case dtype::s16: return mergeScalarNodes<short>(lhs, rhs, op, shape, type);
    case dtype::s32: return mergeScalarNodes<int>(lhs, rhs, op, shape, type);
    case dtype::s64: return mergeScalarNodes<long long>(lhs, rhs, op, shape, type);
    case dtype::u8: return mergeScalarNodes<unsigned char>(lhs, rhs, op, shape, type);
    case dtype::u16: return mergeScalarNodes<unsigned short>(lhs, rhs, op, shape, type);
    case dtype::u32: return mergeScalarNodes<unsigned int>(lhs, rhs, op, shape, type);
    case dtype::u64: return mergeScalarNodes<unsigned long long>(lhs, rhs, op, shape, type);
  }
  throw std::runtime_error("Unknown data type");
}

std::shared_ptr<Node> foldConstants(std::shared_ptr<Node> root);

std::shared_ptr<Node> foldConstantsBinaryNode(std::shared_ptr<Node> root) {
  const auto& binaryRoot = root->impl<BinaryNode>();
  const auto binop = binaryRoot.op();
  const auto newLhs = foldConstants(binaryRoot.lhs());
  const auto newRhs = foldConstants(binaryRoot.rhs());
  if (isNodeUsedInGraphOnly(newLhs) &&
      newLhs->type() == NodeType::Scalar &&
      isNodeUsedInGraphOnly(newRhs) &&
      newRhs->type() == NodeType::Scalar) {
    const auto& lhsScalar = newLhs->impl<ScalarNode>();
    const auto& rhsScalar = newRhs->impl<ScalarNode>();
    const auto shape = lhsScalar.shape();
    const auto dtype = lhsScalar.dataType();
    // TODO casting madness
    if (shape == rhsScalar.shape() && dtype == rhsScalar.dataType()) {
      return mergeScalarNodes(lhsScalar, rhsScalar, binop, shape, dtype);
    }
  }
  // Failed to fold `root`, but if folding happened in either input,
  // we must create a new binary node.
  if (binaryRoot.lhs() != newLhs || binaryRoot.rhs() != newRhs) {
    return BinaryNode::create(newLhs, newRhs, binop);
  }
  return root;
}

std::shared_ptr<Node> foldConstants(std::shared_ptr<Node> root) {
  switch (root->type()) {
    case NodeType::Binary: return foldConstantsBinaryNode(root);
    case NodeType::Custom:
    case NodeType::Scalar:
    case NodeType::Value: return root;
  }
  throw std::runtime_error("Unknown node type");
}

} // namespace

std::shared_ptr<Node> Optimizer::optimize(std::shared_ptr<Node> root) {
  return foldConstants(root);
}

} // namespace fl
