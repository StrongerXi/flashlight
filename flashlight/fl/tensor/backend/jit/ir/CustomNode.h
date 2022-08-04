/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

#include <functional>

namespace fl {

/**
 * A node that represents customized op (e.g., fused tensor operation)
 */
class CustomNode : public NodeTrait<CustomNode> {
 public:
  using EvalFunc = std::function<Tensor(const std::vector<Tensor>&)>;

 private:
  const std::string debugName_;
  const EvalFunc evalFunc_;

  // a trick to enable `std::make_shared` with effecitvely private constructor
  struct PrivateHelper{};

 public:
  static constexpr NodeType nodeType = NodeType::Custom;

  static std::shared_ptr<CustomNode> create(
      std::string&& debugName,
      std::vector<std::shared_ptr<Node>>&& inputs,
      EvalFunc&& evalFunc);

  const std::string& debugName() const;
  const EvalFunc& evalFunc() const;

  std::shared_ptr<Node> mapInputs(
      std::function<std::shared_ptr<Node>(std::shared_ptr<Node>)>&& func
  ) override;

  // intentionally kept unusable publicly
  CustomNode(
      const PrivateHelper&,
      std::string&& debugName,
      std::vector<std::shared_ptr<Node>>&& inputs,
      EvalFunc&& evalFunc);
};

} // namespace fl
