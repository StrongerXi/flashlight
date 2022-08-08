/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"

namespace fl {

ScalarNode::ScalarNode(
    const PrivateHelper&,
    Shape&& shape,
    const fl::dtype type,
    const ScalarType scalar)
    : NodeTrait({}, std::move(shape)), dtype_(type), scalar_(scalar) {}

dtype ScalarNode::dataType() const {
  return dtype_;
}

std::shared_ptr<Node> ScalarNode::mapInputs(
    std::function<std::shared_ptr<Node>(std::shared_ptr<Node>)>&& /* func */) {
  return std::make_shared<ScalarNode>(
      PrivateHelper{}, Shape(shape()), dtype_, scalar_);
}

} // namespace fl
