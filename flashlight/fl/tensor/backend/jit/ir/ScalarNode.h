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
 * TODO node representing scalar of specific shape & type.
 */
class ScalarNode : public NodeTrait<ScalarNode> {
  // these types can hold all types scalars FL support, w/o loss of precision
  using ScalarType = std::variant<long long, double, unsigned long long>;

  const Shape shape_;
  const dtype dtype_;
  const ScalarType scalar_;

  // TODO
  struct PrivateHelper{};

 public:
  static constexpr NodeType nodeType = NodeType::Scalar;

  /**
   * TODO
   */
  template <typename T>
  static std::shared_ptr<ScalarNode> create(
      const Shape& shape,
      const dtype type,
      const T scalar) {
    switch (type) {
      case dtype::b8:
      case dtype::s16:
      case dtype::s32:
      case dtype::s64:
      case dtype::u8:
      case dtype::u16:
      case dtype::u32:
        return std::make_shared<ScalarNode>(
            PrivateHelper{}, shape, type, static_cast<long long>(scalar));
      case dtype::u64:
        return std::make_shared<ScalarNode>(
            PrivateHelper{}, shape, type, static_cast<unsigned long long>(scalar));
      case dtype::f16:
      case dtype::f32:
      case dtype::f64:
        return std::make_shared<ScalarNode>(
            PrivateHelper{}, shape, type, static_cast<double>(scalar));
    }
  }

  // TODO
  ScalarNode(
      const PrivateHelper&,
      const Shape& shape,
      const dtype type,
      const ScalarType scalar);

  /**
   * TODO
   */
  const Shape& shape() const;

  /**
   * TODO
   */
  dtype dataType() const;

  /**
   * TODO
   * cast to T
   */
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
};

} // namespace fl
