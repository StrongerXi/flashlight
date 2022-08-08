/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include <stdexcept>

namespace fl {

BinaryNode::BinaryNode(Node* lhs, Node* rhs, BinaryOp op, const Shape& shape)
    : NodeTrait({lhs, rhs}, shape), op_(op) {}

BinaryNode* BinaryNode::create(Node* lhs, Node* rhs, BinaryOp op) {
    // TODO support broadcast
  if (lhs->shape() != rhs->shape()) {
    throw std::runtime_error(
        "[BinaryNode::BinaryNode] Shape inference doesn't support broadcast yet"
    );
  }
  return new BinaryNode(lhs, rhs, op, lhs->shape());
}

BinaryOp BinaryNode::op() const {
  return op_;
}

Node* BinaryNode::lhs() const {
  return getInput(kLhsIdx);
}

Node* BinaryNode::rhs() const {
  return getInput(kRhsIdx);
}

} // namespace fl
