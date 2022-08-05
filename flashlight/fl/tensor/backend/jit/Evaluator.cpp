/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/Evaluator.h"

#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"

namespace fl {

Evaluator::Evaluator(TensorBackend& backend) : backend_(backend) {}

Tensor Evaluator::evalBinaryNode(BinaryNode& node) {
  const auto& lhs = execute(node.lhs());
  const auto& rhs = execute(node.rhs());
  switch(node.op()) {
    case BinaryOp::Add: return backend_.add(lhs, rhs);
    case BinaryOp::Sub: return backend_.sub(lhs, rhs);
    case BinaryOp::Mul: return backend_.mul(lhs, rhs);
    case BinaryOp::Div: return backend_.div(lhs, rhs);
  }
  throw std::runtime_error("Unknown binary operation type");
}

Tensor Evaluator::evalCustomNode(CustomNode& node) {
  std::vector<Tensor> inputTensors;
  for (auto& inputNode : node.inputs()) {
    inputTensors.emplace_back(evalNode(inputNode));
  }
  return node.evalFunc()(inputTensors);
}

Tensor Evaluator::evalScalarNode(ScalarNode& node) {
  // TODO store original scalar type or cast it to dtype's ctype during node creation
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

Tensor Evaluator::evalNode(std::shared_ptr<Node> node) {
  switch (node->type()) {
    case NodeType::Binary: return evalBinaryNode(node->impl<BinaryNode>());
    case NodeType::Custom: return evalCustomNode(node->impl<CustomNode>());
    case NodeType::Scalar: return evalScalarNode(node->impl<ScalarNode>());
    case NodeType::Value: return node->impl<ValueNode>().value(); // TODO shallow copy
  }
  throw std::runtime_error("Unknown node type");
}

Tensor Evaluator::execute(std::shared_ptr<Node> node) {
  if (!node->getResult().has_value()) {
    node->setResult(evalNode(node));
  }
  return node->getResult().value();
}

} // namespace fl
