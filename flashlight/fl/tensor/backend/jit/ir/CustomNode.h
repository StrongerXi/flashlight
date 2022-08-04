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
 * TODO node representing customized op (e.g., fused tensor operation)
 */
class CustomNode : public NodeTrait<CustomNode> {
  const std::string debugName_;
  const Shape shape_;
  std::function<Tensor(const std::vector<Tensor>&)>&& evalFunc_;

  // TODO
  struct PrivateHelper{};

 public:
  static constexpr NodeType nodeType = NodeType::Custom;

  /**
   * TODO
   */
  static std::shared_ptr<CustomNode> create(
      std::string&& debugName,
      std::vector<std::shared_ptr<Node>>&& inputs,
      const Shape& shape,
      std::function<Tensor(const std::vector<Tensor>&)>&& evalFunc);

  // TODO
  CustomNode(
      const PrivateHelper&,
      std::string&& debugName,
      std::vector<std::shared_ptr<Node>>&& inputs,
      const Shape& shape,
      std::function<Tensor(const std::vector<Tensor>&)>&& evalFunc);

  /**
   * TODO
   */
  const std::string& debugName() const;

  /**
   * TODO
   */
  Shape shape() const;

  /**
   * TODO
   */
  const std::function<Tensor(const std::vector<Tensor>&)>& evalFunc() const;

  // TODO how will data be used?
};

} // namespace fl
