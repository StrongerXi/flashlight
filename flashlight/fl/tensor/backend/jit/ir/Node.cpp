/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

#include <stdexcept>

namespace fl {

void Node::nodeImplTypeCheck(NodeType expect, NodeType actual) const {
  if (expect != actual) {
    std::ostringstream oss;
    oss << "[fl::Node::impl] "
        << "specified node type: [" << actual << "] "
        << "doesn't match actual node type: [" << expect << "]";
    throw std::invalid_argument(oss.str());
  }
}

Node::Node(std::vector<std::shared_ptr<Node>>&& inputs) : inputs_(inputs) {
  for (const auto& inputPtr : inputs_) {
    inputPtr->numNodeUsers_++;
  }
}

Node::~Node() {
  for (const auto& inputPtr : inputs_) {
    inputPtr->numNodeUsers_--;
  }
}

std::shared_ptr<Node> Node::getInput(unsigned idx) const {
  return inputs_.at(idx);
}

const std::vector<std::shared_ptr<Node>>& Node::inputs() const {
  return inputs_;
}

const std::optional<Tensor>& Node::getResult() const {
  return result_;
}

void Node::setResult(Tensor&& tensor) {
  if (result_.has_value()) {
    throw std::invalid_argument("[Node::setResult] Result already set");
  } else {
    result_ = std::move(tensor);
  }
}

unsigned Node::numNodeUsers() const {
  return numNodeUsers_;
}

void Node::replaceInputNode(
    const std::shared_ptr<Node>& oldInput,
    const std::shared_ptr<Node>& newInput) {
  for (unsigned i = 0; i < inputs_.size(); i++) {
    if (inputs_[i] == oldInput) {
      inputs_[i]->numNodeUsers_--;
      inputs_[i] = newInput;
    }
  }
}

} // namespace fl
