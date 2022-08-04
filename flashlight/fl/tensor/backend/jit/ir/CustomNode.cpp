/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/CustomNode.h"
#include <vector>

namespace fl {

CustomNode::CustomNode(
    const PrivateHelper&,
    std::string&& debugName,
    std::vector<std::shared_ptr<Node>>&& inputs,
    EvalFunc&& evalFunc)
    : NodeTrait(std::move(inputs)),
    debugName_(debugName), evalFunc_(std::move(evalFunc)) {}


std::shared_ptr<CustomNode> CustomNode::create(
    std::string&& debugName,
    std::vector<std::shared_ptr<Node>>&& inputs,
    EvalFunc&& evalFunc) {
  return std::make_shared<CustomNode>(
          PrivateHelper{},
          std::move(debugName),
          std::move(inputs),
          std::move(evalFunc));
}

const std::string& CustomNode::debugName() const {
  return debugName_;
}

const CustomNode::EvalFunc& CustomNode::evalFunc() const {
  return evalFunc_;
}

std::shared_ptr<Node> CustomNode::mapInputs(
    std::function<std::shared_ptr<Node>(std::shared_ptr<Node>)>&& func) {
  std::vector<std::shared_ptr<Node>> newInputs;
  for (const auto& oldInput : inputs()) {
      newInputs.emplace_back(func(oldInput));
  }
  return create(
      std::string(debugName_), std::move(newInputs), EvalFunc(evalFunc_));
}

} // namespace fl
