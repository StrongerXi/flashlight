/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/eval/Evaluator.h"

#include <chrono>
#include <functional>
#include <iostream>

#include "flashlight/fl/tensor/backend/jit/JitTensorBase.h"
#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"
#include "flashlight/fl/tensor/backend/jit/printer/GraphvizPrinter.h"

namespace fl {

Evaluator::Evaluator(TensorBackend& backend) : backend_(backend) {}

template <typename T>
T Evaluator::profile(std::function<T()> func, const Node* nodePtr) {
  const auto start = std::chrono::high_resolution_clock::now();
  auto res = func();
  const auto end = std::chrono::high_resolution_clock::now();
  const auto durNs =
    std::chrono::duration_cast<std::chrono::duration<float>>(end -
        start).count();
  nodeToTotTimeMs_.insert({nodePtr, durNs * 1000});
  return res;
}

const Tensor Evaluator::evalBinaryOp(
    const Tensor& lhs, const Tensor& rhs, BinaryOp op) {
  switch(op) {
    case BinaryOp::Add: return backend_.add(lhs, rhs);
    case BinaryOp::Sub: return backend_.sub(lhs, rhs);
    case BinaryOp::Mul: return backend_.mul(lhs, rhs);
    case BinaryOp::Div: return backend_.div(lhs, rhs);
    case BinaryOp::Eq: return backend_.eq(lhs, rhs);
    case BinaryOp::Neq: return backend_.neq(lhs, rhs);
    case BinaryOp::Gt: return backend_.greaterThan(lhs, rhs);
    case BinaryOp::Gte: return backend_.greaterThanEqual(lhs, rhs);
    case BinaryOp::Lt: return backend_.lessThan(lhs, rhs);
    case BinaryOp::Lte: return backend_.lessThanEqual(lhs, rhs);
  }
  throw std::runtime_error("Unknown binary operation type");
}

const Tensor Evaluator::evalBinaryNode(BinaryNode& node) {
  const auto lhs = getTensorOrEvalNode(node.lhs());
  const auto rhs = getTensorOrEvalNode(node.rhs());
  std::function<const Tensor()> func = [&]() {
    return evalBinaryOp(lhs, rhs, node.op());
  };
  return profile(func, &node);
}

const Tensor Evaluator::evalCustomNode(CustomNode& node) {
  std::vector<Tensor> inputTensors;
  for (auto& inputNode : node.inputs()) {
    inputTensors.emplace_back(getTensorOrEvalNode(inputNode).shallowCopy());
  }
  std::function<const Tensor()> func = [&]() {
    return node.evalFunc()(inputTensors);
  };
  return profile(func, &node);
}

const Tensor Evaluator::evalIndexNode(IndexNode& node) {
  const auto indexedTensor = getTensorOrEvalNode(node.indexedNode());
  std::function<const Tensor()> func = [&]() {
    // NOTE we count indices evaluation too because it's not in the graph
    const auto evaluatedIndices = evalIndices(node.indices());
    return indexedTensor(evaluatedIndices);
  };
  return profile(func, &node);
}

const Tensor Evaluator::evalIndexedMergeNode(IndexedMergeNode& node) {
  // TODO no need to copy if indexedNode has only 1 user here
  const auto indexedTensor = getTensorOrEvalNode(node.indexedNode()).copy();
  const auto mergeSourceTensor = getTensorOrEvalNode(node.mergeSourceNode());
  std::function<const Tensor()> func = [&]() {
    // NOTE we count indices evaluation too because it's not in the graph
    const auto evaluatedIndices = evalIndices(node.indices());
    indexedTensor(evaluatedIndices) = mergeSourceTensor;
    return indexedTensor;
  };
  return profile(func, &node);
}

std::vector<Index> Evaluator::evalIndices(const std::vector<Index>& indices) {
  std::vector<Index> evaluatedIndices;
  for (const auto& index : indices) {
    if (index.type() == detail::IndexType::Tensor) {
      const auto tensorIndexNode = toJitTensorBase(index.get<Tensor>()).node();
      evaluatedIndices.push_back(getTensorOrEvalNode(tensorIndexNode));
    } else {
      evaluatedIndices.push_back(index);
    }
  }
  return evaluatedIndices;
}

const Tensor Evaluator::evalScalarNode(ScalarNode& node) {
  std::function<const Tensor()> func = [&]() {
    const Shape& shape = node.shape();
    const auto dtype = node.dataType();
    switch (dtype) {
      case dtype::f16:
        throw std::invalid_argument(
            "[JitTensor::evalScalarNode] dtype::f16 not supported");
      case dtype::f32:
        return backend_.full(shape, node.scalar<float>(), dtype);
      case dtype::f64:
        return backend_.full(shape, node.scalar<double>(), dtype);
      case dtype::b8:
        return backend_.full(shape, node.scalar<char>(), dtype);
      case dtype::s16:
        return backend_.full(shape, node.scalar<short>(), dtype);
      case dtype::s32:
        return backend_.full(shape, node.scalar<int>(), dtype);
      case dtype::s64:
        return backend_.full(shape, node.scalar<long long>(), dtype);
      case dtype::u8:
        return backend_.full(shape, node.scalar<unsigned char>(), dtype);
      case dtype::u16:
        return backend_.full(shape, node.scalar<unsigned short>(), dtype);
      case dtype::u32:
        return backend_.full(shape, node.scalar<unsigned int>(), dtype);
      case dtype::u64:
        return backend_.full(shape, node.scalar<unsigned long long>(), dtype);
    }
    throw std::runtime_error("Unknown dtype");
  };
  return profile(func, &node);
}

const Tensor Evaluator::getTensorOrEvalNode(std::shared_ptr<Node> node) {
  auto evalNode = [this](std::shared_ptr<Node> node) {
    switch (node->type()) {
      case NodeType::Binary: return evalBinaryNode(node->impl<BinaryNode>());
      case NodeType::Custom: return evalCustomNode(node->impl<CustomNode>());
      case NodeType::Index: return evalIndexNode(node->impl<IndexNode>());
      case NodeType::IndexedMerge:
          return evalIndexedMergeNode(node->impl<IndexedMergeNode>());
      case NodeType::Scalar: return evalScalarNode(node->impl<ScalarNode>());
      case NodeType::Value:
        // read-only shallow copy
        return node->impl<ValueNode>().value().shallowCopy();
    }
    throw std::runtime_error("Unknown node type");
  };

  const auto& optResult = node->getResult();
  if (!node->getResult().has_value()) {
    // read-only shallow copy
    node->setResult(evalNode(node).shallowCopy());
  }
  // read-only shallow copy
  return optResult.value().shallowCopy();
}

void Evaluator::execute(std::shared_ptr<Node> node) {
  static GraphvizPrinter printer;
  if (!node->getResult().has_value()) {
    getTensorOrEvalNode(node);
    printer.printSubgraph(node, std::cout, "eval", nodeToTotTimeMs_);
    nodeToTotTimeMs_.clear();
  }
}

} // namespace fl
