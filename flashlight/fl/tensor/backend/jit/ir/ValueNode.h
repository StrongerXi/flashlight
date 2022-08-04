/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

namespace fl {

/**
 * TODO node representing an evaluated tensor
 */
class ValueNode : public NodeTrait<ValueNode> {
  // TODO
  struct PrivateHelper{};

 public:
  static constexpr NodeType nodeType = NodeType::Value;

  /**
   * TODO
   */
  static std::shared_ptr<ValueNode> create(Tensor&& value);

  // TODO
  ValueNode(const PrivateHelper&, Tensor&& value);

  /**
   * TODO
   */
  const Tensor& value() const;
};

} // namespace fl
