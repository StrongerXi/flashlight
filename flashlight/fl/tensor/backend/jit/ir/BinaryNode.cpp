/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"

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
    BinaryOp op) : NodeTrait({ lhs, rhs }), op_(op) {}

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
  return create(func(lhs()), func(rhs()), op_);
}

} // namespace fl
