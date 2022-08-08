/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/opt/Optimizer.h"

#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorExtension.h"
#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/CustomNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/IndexNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"
#include "flashlight/fl/tensor/backend/jit/opt/JitOptimizerExtension.h"
#include "flashlight/fl/tensor/backend/jit/opt/JitOptimizerExtensionBackends.h"

namespace fl {

namespace {

template<typename T>
T foldScalars(const T lhs, const T rhs, const BinaryOp op) {
  switch (op) {
    case BinaryOp::Add: return lhs + rhs;
    case BinaryOp::Sub: return lhs - rhs;
    case BinaryOp::Mul: return lhs * rhs;
    case BinaryOp::Div: return lhs / rhs;
  }
  throw std::runtime_error("Unknown binary operation type");
}

template<typename T>
std::shared_ptr<ScalarNode> foldScalarNodes(
    const ScalarNode& lhs,
    const ScalarNode& rhs,
    const BinaryOp op,
    const dtype type) {
  T lhsVal = lhs.scalar<T>();
  T rhsVal = rhs.scalar<T>();
  T resVal = foldScalars(lhsVal, rhsVal, op);
  return ScalarNode::create(Shape(lhs.shape()), type, resVal);
}

std::shared_ptr<ScalarNode> foldScalarNodes(
    const ScalarNode& lhs,
    const ScalarNode& rhs,
    const BinaryOp op,
    const dtype type) {
  switch (type) {
    case dtype::f16:
      throw std::runtime_error("[foldScalarNodes] f16 is unsupported");
    case dtype::f32: return foldScalarNodes<float>(lhs, rhs, op, type);
    case dtype::f64: return foldScalarNodes<double>(lhs, rhs, op, type);
    case dtype::b8: return foldScalarNodes<char>(lhs, rhs, op, type);
    case dtype::s16: return foldScalarNodes<short>(lhs, rhs, op, type);
    case dtype::s32: return foldScalarNodes<int>(lhs, rhs, op, type);
    case dtype::s64: return foldScalarNodes<long long>(lhs, rhs, op, type);
    case dtype::u8: return foldScalarNodes<unsigned char>(lhs, rhs, op, type);
    case dtype::u16: return foldScalarNodes<unsigned short>(lhs, rhs, op, type);
    case dtype::u32: return foldScalarNodes<unsigned int>(lhs, rhs, op, type);
    case dtype::u64:
      return foldScalarNodes<unsigned long long>(lhs, rhs, op, type);
  }
  throw std::runtime_error("Unknown data type");
}

std::shared_ptr<Node> foldScalars(std::shared_ptr<Node> node);

std::shared_ptr<Node> foldScalarsInBinaryNode(std::shared_ptr<Node> node) {
  const auto& binaryNode = node->impl<BinaryNode>();
  const auto binop = binaryNode.op();
  const auto oldLhs = binaryNode.lhs();
  const auto oldRhs = binaryNode.rhs();
  const auto newLhs = foldScalars(oldLhs);
  const auto newRhs = foldScalars(oldRhs);
  // no need to check `mustMaterialize`, because if we can fold scalars, the
  // net profit is almost (always?) positive.
  if (newLhs->isScalar() && newRhs->isScalar()) {
    const auto& lhsScalar = newLhs->impl<ScalarNode>();
    const auto& rhsScalar = newRhs->impl<ScalarNode>();
    const auto& shape = lhsScalar.shape();
    const auto dtype = lhsScalar.dataType();
    if (shape == rhsScalar.shape() && dtype == rhsScalar.dataType()) {
      return foldScalarNodes(lhsScalar, rhsScalar, binop, dtype);
    }
  }
  // Failed to fold `node`, but if folding happened in either input,
  // we should create a new binary node to benefit from that -- no mutation yet.
  if (oldLhs != newLhs || oldRhs != newRhs) {
    return BinaryNode::create(newLhs, newRhs, binop);
  }
  return node;
}

std::shared_ptr<Node> foldScalars(std::shared_ptr<Node> node) {
  // TODO
  // 1. go crazy -- we can even optimize the (Jit)Tensor indices.
  // 2. time for a visitor to take care of
  //    - "not create new node if inputs didn't change"
  //    - traversing nodes we don't care about by default
  switch (node->type()) {
    case NodeType::Binary: return foldScalarsInBinaryNode(node);
    case NodeType::Index:
    case NodeType::Custom:
    case NodeType::Scalar:
    case NodeType::Value:
      return node->mapInputs([](auto input) { return foldScalars(input); });
  }
  throw std::runtime_error("Unknown node type");
}

} // namespace

Optimizer::Optimizer(TensorBackend& backend) : backend_(backend) {}

std::shared_ptr<Node> Optimizer::optimize(std::shared_ptr<Node> node) {
  node = foldScalars(node);
  auto& registrar = detail::TensorExtensionRegistrar::getInstance();
  if (registrar.isTensorExtensionRegistered(
        backend_.backendType(), TensorExtensionType::JitOptimizer)) {
    node = backend_.getExtension<JitOptimizerExtension>().optimize(node);
  }
  return node;
}

} // namespace fl
