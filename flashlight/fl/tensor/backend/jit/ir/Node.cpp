/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

#include <sstream>
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
  for (const auto& input : inputs) {
    input->incUseCount();
  }
}

Node::~Node() {
  for (const auto& input : inputs()) {
    input->decUseCount();
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

unsigned Node::getUseCount() const {
  return useCount_;
}

void Node::incUseCount() {
  useCount_++; // this can't possibly overflow...
}

void Node::decUseCount() {
  if (useCount_ == 0) {
    throw std::runtime_error("[Node::decUseCount] Currently no uses");
  }
  useCount_--;
}

bool Node::isBinary() const {
  return type() == NodeType::Binary;
}

bool Node::isCustom() const {
  return type() == NodeType::Custom;
}

bool Node::isScalar() const {
  return type() == NodeType::Scalar;
}

bool Node::isValue() const {
  return type() == NodeType::Value;
}

} // namespace fl
