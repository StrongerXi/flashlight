/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include <stdexcept>

namespace fl {

std::shared_ptr<BinaryNode> BinaryNode::create(
    std::shared_ptr<Node> lhs,
    std::shared_ptr<Node> rhs,
    BinaryOp op) {
  return std::make_shared<BinaryNode>(PrivateHelper{}, lhs, rhs, op);
}

BinaryNode::BinaryNode(
    const PrivateHelper&,
    std::shared_ptr<Node> lhs,
    std::shared_ptr<Node> rhs,
    BinaryOp op) : NodeTrait({ lhs, rhs }, Shape(lhs->shape())), op_(op) {
  // TODO resolve broadcast shape
  if (lhs->shape() != rhs->shape()) {
    throw std::runtime_error(
        "[BinaryNode::BinaryNode] Shape inference doesn't support broadcast yet"
    );
  }
}

BinaryOp BinaryNode::op() const {
  return op_;
}

std::shared_ptr<Node> BinaryNode::lhs() const {
  return getInput(kLhsIdx);
}

std::shared_ptr<Node> BinaryNode::rhs() const {
  return getInput(kRhsIdx);
}

std::shared_ptr<Node> BinaryNode::mapInputs(
    std::function<std::shared_ptr<Node>(std::shared_ptr<Node>)>&& func) {
  const auto newLhs = func(lhs());
  const auto newRhs = func(rhs());
  if (newLhs == lhs() && newRhs == rhs()) {
    return shared_from_this();
  }
  return create(newLhs, newRhs, op_);
}

} // namespace fl
