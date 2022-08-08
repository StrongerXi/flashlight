/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"

namespace fl {

ValueNode::ValueNode(const PrivateHelper&, Tensor&& value)
  : NodeTrait({}, Shape(value.shape())) {
  setResult(std::move(value));
}


std::shared_ptr<ValueNode> ValueNode::create(Tensor&& value) {
  return std::make_shared<ValueNode>(PrivateHelper{}, std::move(value));
}

const Tensor& ValueNode::value() const {
  return getResult().value(); // guaranteed to be there
}

std::shared_ptr<Node> ValueNode::mapInputs(
    std::function<std::shared_ptr<Node>(std::shared_ptr<Node>)>&& /* func */) {
  // no inputs, just return this
  // NOTE technically we could shallowCopy tensor here, since Tensors in the
  // graph are immutable, but that requires friending `Tensor` and ugh...
  return shared_from_this();
}

} // namespace fl
