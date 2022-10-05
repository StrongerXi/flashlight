/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/eval/Evaluator.h"

#include <chrono>
#include <functional>
#include <queue>
#include <iostream>

#include "flashlight/fl/tensor/backend/jit/JitTensorBase.h"
#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"
#include "flashlight/fl/tensor/backend/jit/printer/GraphvizPrinter.h"

namespace fl {

namespace {

// Build a map from each node in the tree to its current refcount.
std::unordered_map<Node*, unsigned> getNodeToRefCountInTree(Node* root) {
  std::unordered_map<Node*, unsigned> nodeToRefCount;
  std::queue<Node*> worklist({root}); // nodes to be visited
  while (!worklist.empty()) {
    Node* node = worklist.front();
    worklist.pop();
    if (nodeToRefCount.find(node) == nodeToRefCount.end()) {
      nodeToRefCount.emplace(node, node->getRefCount());
      for (const auto& input : node->inputs()) {
        worklist.push(input);
      }
    }
  }
  return nodeToRefCount;
}

} // namespace

Evaluator::Evaluator(TensorBackend& backend) : backend_(backend) {}

void Evaluator::profile(std::function<void()> func, Node* nodePtr) {
  const auto start = std::chrono::high_resolution_clock::now();
  func();
  const auto end = std::chrono::high_resolution_clock::now();
  const auto durNs =
    std::chrono::duration_cast<std::chrono::duration<float>>(end -
        start).count();
  nodeToTotTimeMs_.insert({nodePtr, durNs * 1000});
}

void Evaluator::evalBinaryNode(BinaryNode& node) {
  evalNode(node.lhs());
  evalNode(node.rhs());
  const auto& lhs = node.lhs()->getResult().value();
  const auto& rhs = node.rhs()->getResult().value();
  std::function<void()> func = [&]() {
    node.setResult(evalBinaryOp(node.op(), lhs, rhs));
  };
  profile(func, &node);
}

void Evaluator::evalCustomNode(CustomNode& node) {
  std::vector<const Tensor*> inputTensors;
  for (auto& inputNode : node.inputs()) {
    evalNode(inputNode);
    inputTensors.push_back(&inputNode->getResult().value());
  }
  std::function<void()> func = [&]() {
    node.setResult(node.evalFunc()(inputTensors));
  };
  profile(func, &node);
}

void Evaluator::evalIndexNode(IndexNode& node) {
  evalNode(node.indexedNode());
  const auto& indexedTensor = node.indexedNode()->getResult().value();
  std::function<void()> func = [&]() {
    // NOTE we count indices evaluation too because they are not in the graph
    // (tensor index is not an "input" to the IndexNode)
    node.setResult(indexedTensor(evalIndices(node.indices())));
  };
  profile(func, &node);
}

void Evaluator::evalIndexedMergeNode(IndexedMergeNode& node) {
  evalNode(node.indexedNode());
  // TODO no need to copy if indexedNode has only 1 user here
  auto indexedTensor = node.indexedNode()->getResult().value().copy();
  evalNode(node.mergeSourceNode());
  const auto& mergeSourceTensor = node.mergeSourceNode()->getResult().value();
  std::function<void()> func = [&]() {
    // NOTE we count indices evaluation too because they are not in the graph
    // (tensor index is not an "input" to the IndexNode)
    const auto evaluatedIndices = evalIndices(node.indices());
    indexedTensor(evaluatedIndices) = mergeSourceTensor;
    node.setResult(std::move(indexedTensor));
  };
  profile(func, &node);
}

std::vector<Index> Evaluator::evalIndices(const std::vector<Index>& indices) {
  std::vector<Index> evaluatedIndices;
  for (const auto& index : indices) {
    if (index.type() == detail::IndexType::Tensor) {
      const auto tensorIndexNode = toJitTensorBase(index.get<Tensor>()).node();
      evalNode(tensorIndexNode);
      evaluatedIndices.push_back(tensorIndexNode->getResult().value());
    } else {
      evaluatedIndices.push_back(index);
    }
  }
  return evaluatedIndices;
}

void Evaluator::evalScalarNode(ScalarNode& node) {
  std::function<void()> func = [&]() {
    node.setResult(evalScalar(node));
  };
  profile(func, &node);
}

Tensor
Evaluator::evalBinaryOp(BinaryOp op, const Tensor& lhs, const Tensor& rhs) {
  switch (op) {
    case BinaryOp::Add:
      return backend_.add(lhs, rhs);
    case BinaryOp::Sub:
      return backend_.sub(lhs, rhs);
    case BinaryOp::Mul:
      return backend_.mul(lhs, rhs);
    case BinaryOp::Div:
      return backend_.div(lhs, rhs);
    case BinaryOp::Eq:
      return backend_.eq(lhs, rhs);
    case BinaryOp::Neq:
      return backend_.neq(lhs, rhs);
    case BinaryOp::Gt:
      return backend_.greaterThan(lhs, rhs);
    case BinaryOp::Gte:
      return backend_.greaterThanEqual(lhs, rhs);
    case BinaryOp::Lt:
      return backend_.lessThan(lhs, rhs);
    case BinaryOp::Lte:
      return backend_.lessThanEqual(lhs, rhs);
  }
  throw std::runtime_error(
      "[Evaluator::evalBinaryOp] Unknown binary operation type");
}

Tensor Evaluator::evalScalar(ScalarNode& node) {
  //const Shape& shape = node.shape();
  Shape shape(std::vector<Dim>(node.shape().ndim(), 1));
  const auto dtype = node.dataType();
  switch (dtype) {
    case dtype::f16:
    case dtype::f32:
    case dtype::f64:
      return backend_.full(shape, node.scalar<double>(), dtype);
    case dtype::b8:
    case dtype::s16:
    case dtype::s32:
    case dtype::s64:
    case dtype::u8:
    case dtype::u16:
    case dtype::u32:
      return backend_.full(shape, node.scalar<long long>(), dtype);
    case dtype::u64:
      return backend_.full(shape, node.scalar<unsigned long long>(), dtype);
  }
  throw std::runtime_error("Unknown dtype");
}

void Evaluator::evalNodeDispatch(Node* node) {
  switch (node->type()) {
    case NodeType::Binary:
      return evalBinaryNode(node->impl<BinaryNode>());
    case NodeType::Custom:
      return evalCustomNode(node->impl<CustomNode>());
    case NodeType::Index:
      return evalIndexNode(node->impl<IndexNode>());
    case NodeType::IndexedMerge:
      return evalIndexedMergeNode(node->impl<IndexedMergeNode>());
    case NodeType::Scalar:
      return evalScalarNode(node->impl<ScalarNode>());
    case NodeType::Value:
      return; // already has a result
  }
  throw std::runtime_error("[Evaluator::evalNodeDispatch] Unknown node type");
}

void Evaluator::evalNode(Node* node) {
  if (!node->getResult().has_value()) {
    evalNodeDispatch(node);
    for (const auto& input : node->inputs()) {
      auto& count = nodeToResultUseCount_.at(input);
      count--;
      if (count == 0) {
        // This helps reduce memory footprint during evaluation, allowing the
        // result tensor memory to be reused. This has a non-trivial performance
        // impact on graph with high intermediate tensor memory usage.
        input->unsetResult();
      }
    }
  }
}

void Evaluator::eval(Node* node) {
  static GraphvizPrinter printer;
  nodeToResultUseCount_ = getNodeToRefCountInTree(node);
  evalNode(node);
  //printer.printSubgraph(node, std::cout, "eval", nodeToTotTimeMs_);
  nodeToTotTimeMs_.clear();
  nodeToResultUseCount_.clear();
}

} // namespace fl
