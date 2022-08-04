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
 * TODO types of binary operations.
 */
enum class BinaryOp {
  Add,
  Sub,
  Mul,
  Div
};

/**
 * TODO node representing binary operations.
 */
class BinaryNode : public NodeTrait<BinaryNode> {
  const BinaryOp op_;

  // TODO
  struct PrivateHelper{};

  // TODO
  static constexpr unsigned kLhsIdx = 0;
  static constexpr unsigned kRhsIdx = 1;

 public:
  static constexpr NodeType nodeType = NodeType::Binary;

  /**
   * TODO
   */
  static std::shared_ptr<BinaryNode> create(
      std::shared_ptr<Node> lhs,
      std::shared_ptr<Node> rhs,
      BinaryOp op);

  // TODO
  BinaryNode(
      const PrivateHelper&,
      std::shared_ptr<Node> lhs,
      std::shared_ptr<Node> rhs,
      BinaryOp op);

  /**
   * TODO
   */
  BinaryOp op() const;

  /**
   * TODO
   */
  std::shared_ptr<Node> lhs() const;

  /**
   * TODO
   */
  std::shared_ptr<Node> rhs() const;
};

} // namespace fl
