/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

namespace fl {

/**
 * Types of binary operations.
 */
enum class BinaryOp {
  Add,
  Sub,
  Mul,
  Div,
  Eq,
  Neq,
  Gt,
  Gte,
  Lt,
  Lte,
};

/**
 * A node that represents binary operations.
 */
class BinaryNode : public NodeTrait<BinaryNode> {
  const BinaryOp op_;

  // a trick to enable `std::make_shared` with effecitvely private constructor
  struct PrivateHelper{};

  // helps indexing into inputs
  static constexpr unsigned kLhsIdx = 0;
  static constexpr unsigned kRhsIdx = 1;

 public:
  static constexpr NodeType nodeType = NodeType::Binary;

  static std::shared_ptr<BinaryNode> create(
      std::shared_ptr<Node> lhs,
      std::shared_ptr<Node> rhs,
      BinaryOp op);

  BinaryOp op() const;
  std::shared_ptr<Node> lhs() const;
  std::shared_ptr<Node> rhs() const;

  std::shared_ptr<Node> mapInputs(
      std::function<std::shared_ptr<Node>(std::shared_ptr<Node>)>&& func
  ) override;

  // intentionally kept unusable publicly
  BinaryNode(
      const PrivateHelper&,
      std::shared_ptr<Node> lhs,
      std::shared_ptr<Node> rhs,
      BinaryOp op);
};

} // namespace fl
