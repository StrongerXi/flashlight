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

#include <variant>

namespace fl {

/**
 * A node that represents scalar of specific shape & type.
 */
class ScalarNode :
  public NodeTrait<ScalarNode>,
  public std::enable_shared_from_this<ScalarNode> {
  // these types can hold all types scalars FL support, w/o loss of precision
  using ScalarType = std::variant<long long, double, unsigned long long>;

  const dtype dtype_;
  const ScalarType scalar_; // value used for initialization

  // a trick to enable `std::make_shared` with effecitvely private constructor
  struct PrivateHelper{};

 public:
  static constexpr NodeType nodeType = NodeType::Scalar;

  template <typename T>
  static std::shared_ptr<ScalarNode> create(
      Shape&& shape,
      const dtype type,
      const T scalar) {
    switch (type) {
      case dtype::b8:
      case dtype::s16:
      case dtype::s32:
      case dtype::s64:
      case dtype::u8:
      case dtype::u16:
      case dtype::u32: {
        auto castedScalar = static_cast<long long>(scalar);
        return std::make_shared<ScalarNode>(
            PrivateHelper{}, std::move(shape), type, castedScalar);
      }
      case dtype::u64: {
        auto castedScalar = static_cast<unsigned long long>(scalar);
        return std::make_shared<ScalarNode>(
            PrivateHelper{}, std::move(shape), type, castedScalar);
      }
      case dtype::f16:
      case dtype::f32:
      case dtype::f64:
        auto castedScalar = static_cast<double>(scalar);
        return std::make_shared<ScalarNode>(
            PrivateHelper{}, std::move(shape), type, castedScalar);
    }
    throw std::runtime_error("Unknown dtype");
  }

  // metadata
  dtype dataType() const;

  // cast to T
  template <typename T>
  T scalar() const {
    // TODO once we have full c++17 support, use `std::visit(overloaded {...})`
    if (std::holds_alternative<long long>(scalar_)) {
      return std::get<long long>(scalar_);
    } else if (std::holds_alternative<unsigned long long>(scalar_)) {
      return std::get<unsigned long long>(scalar_);
    } else if (std::holds_alternative<double>(scalar_)) {
      return std::get<double>(scalar_);
    }
    throw std::runtime_error("Unknown scalar variant");
  }

  std::shared_ptr<Node> mapInputs(
      std::function<std::shared_ptr<Node>(std::shared_ptr<Node>)>&& func
  ) override;

  // intentionally kept unusable publicly
  ScalarNode(
      const PrivateHelper&,
      Shape&& shape,
      const dtype type,
      const ScalarType scalar);
};

} // namespace fl
