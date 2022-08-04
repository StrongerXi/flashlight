/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"

namespace fl {

ValueNode::ValueNode(const PrivateHelper&, Tensor&& value) : NodeTrait({}) {
  setResult(std::move(value));
}


std::shared_ptr<ValueNode> ValueNode::create(Tensor&& value) {
  return std::make_shared<ValueNode>(PrivateHelper{}, std::move(value));
}

const Tensor& ValueNode::value() const {
  return getResult().value(); // guaranteed to be there
}

} // namespace fl
