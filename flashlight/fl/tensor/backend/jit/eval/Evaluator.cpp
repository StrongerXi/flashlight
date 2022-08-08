/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/eval/Evaluator.h"

#include <queue>

#include "flashlight/fl/tensor/backend/jit/JitTensorBase.h"
#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"

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

void Evaluator::evalBinaryNode(BinaryNode& node) {
  evalNode(node.lhs());
  evalNode(node.rhs());
  const auto& lhs = node.lhs()->getResult().value();
  const auto& rhs = node.rhs()->getResult().value();
  node.setResult(evalBinaryOp(node.op(), lhs, rhs));
}

void Evaluator::evalCustomNode(CustomNode& node) {
  std::vector<const Tensor*> inputTensors;
  for (auto& inputNode : node.inputs()) {
    evalNode(inputNode);
    inputTensors.push_back(&inputNode->getResult().value());
  }
  node.setResult(node.evalFunc()(inputTensors));
}

void Evaluator::evalIndexNode(IndexNode& node) {
  evalNode(node.indexedNode());
  const auto& indexedTensor = node.indexedNode()->getResult().value();
  node.setResult(indexedTensor(evalIndices(node.indices())));
}

void Evaluator::evalIndexedMergeNode(IndexedMergeNode& node) {
  evalNode(node.indexedNode());
  // TODO no need to copy if indexedNode has only 1 user here
  auto indexedTensor = node.indexedNode()->getResult().value().copy();
  const auto evaluatedIndices = evalIndices(node.indices());
  evalNode(node.mergeSourceNode());
  const auto& mergeSourceTensor = node.mergeSourceNode()->getResult().value();
  indexedTensor(evaluatedIndices) = mergeSourceTensor;
  node.setResult(std::move(indexedTensor));
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
  node.setResult(evalScalar(node));
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
  }
  throw std::runtime_error(
      "[Evaluator::evalBinaryOp] Unknown binary operation type");
}

Tensor Evaluator::evalScalar(ScalarNode& node) {
  const Shape& shape = node.shape();
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
  nodeToResultUseCount_ = getNodeToRefCountInTree(node);
  evalNode(node);
  nodeToResultUseCount_.clear();
}

} // namespace fl
