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
 * A node that represents an evaluated tensor
 */
class ValueNode :
  public NodeTrait<ValueNode>,
  public std::enable_shared_from_this<ValueNode> {
  // a trick to enable `std::make_shared` with effecitvely private constructor
  struct PrivateHelper{};

 public:
  static constexpr NodeType nodeType = NodeType::Value;

  static std::shared_ptr<ValueNode> create(Tensor&& value);
  const Tensor& value() const;

  std::shared_ptr<Node> mapInputs(
      std::function<std::shared_ptr<Node>(std::shared_ptr<Node>)>&& func
  ) override;

  // intentionally kept unusable publicly
  ValueNode(const PrivateHelper&, Tensor&& value);
};

} // namespace fl
