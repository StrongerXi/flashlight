/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/CustomNode.h"

namespace fl {

CustomNode::CustomNode(
    const PrivateHelper&,
    std::string&& debugName,
    std::vector<std::shared_ptr<Node>>&& inputs,
    const Shape& shape,
    std::function<Tensor(const std::vector<Tensor>&)>&& evalFunc)
    : NodeTrait(std::move(inputs)),
    debugName_(debugName), shape_(shape), evalFunc_(std::move(evalFunc)) {}


std::shared_ptr<CustomNode> CustomNode::create(
    std::string&& debugName,
    std::vector<std::shared_ptr<Node>>&& inputs,
    const Shape& shape,
    std::function<Tensor(const std::vector<Tensor>&)>&& evalFunc) {
  return std::make_shared<CustomNode>(
          PrivateHelper{},
          std::move(debugName),
          std::move(inputs),
          shape,
          std::move(evalFunc));
}

const std::string& CustomNode::debugName() const {
  return debugName_;
}

Shape CustomNode::shape() const {
  return shape_;
}

const std::function<Tensor(const std::vector<Tensor>&)>& CustomNode::evalFunc() const {
  return evalFunc_;
}

} // namespace fl
