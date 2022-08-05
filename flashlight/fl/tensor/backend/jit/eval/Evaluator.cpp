/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/eval/Evaluator.h"

#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"

namespace fl {

Evaluator::Evaluator(TensorBackend& backend) : backend_(backend) {}

const Tensor Evaluator::evalBinaryNode(BinaryNode& node) {
  const auto lhs = getTensorOrEvalNode(node.lhs());
  const auto rhs = getTensorOrEvalNode(node.rhs());
  switch(node.op()) {
    case BinaryOp::Add: return backend_.add(lhs, rhs);
    case BinaryOp::Sub: return backend_.sub(lhs, rhs);
    case BinaryOp::Mul: return backend_.mul(lhs, rhs);
    case BinaryOp::Div: return backend_.div(lhs, rhs);
  }
  throw std::runtime_error("Unknown binary operation type");
}

const Tensor Evaluator::evalCustomNode(CustomNode& node) {
  std::vector<Tensor> inputTensors;
  for (auto& inputNode : node.inputs()) {
    inputTensors.emplace_back(getTensorOrEvalNode(inputNode));
  }
  return node.evalFunc()(inputTensors);
}

const Tensor Evaluator::evalScalarNode(ScalarNode& node) {
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
}

const Tensor Evaluator::getTensorOrEvalNode(std::shared_ptr<Node> node) {
  auto evalNode = [this](std::shared_ptr<Node> node) {
    switch (node->type()) {
      case NodeType::Binary: return evalBinaryNode(node->impl<BinaryNode>());
      case NodeType::Custom: return evalCustomNode(node->impl<CustomNode>());
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
  getTensorOrEvalNode(node);
}

} // namespace fl
