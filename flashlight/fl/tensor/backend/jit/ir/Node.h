/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/jit/ir/NodeType.h"

namespace fl {

/**
 * A Node that represents Tensor ops.
 *
 * Conceptually, nodes implicitly form a computation DAG with
 * - only user-->used edges
 * - immutable edges
 * - immutable result (i.e., evaluation should be idemponent)
 *
 * To express mutation/alias, think SSA.
 *
 * TODO
 * 1. Consider graph mutability, so that
 *  (a). optimization/rewrite of a node benefit all node users, not just the one
 *       that initiated the optimization.
 *
 * 2. Consider keeping track of used-->user edges (e.g., `userNodes()`)
 *  (a). This helps enable graph-rewrite. Right now the Jit tensor backend can
 *       only leverage Optimizer to do tree-rewrite, because it's initiated from
 *       a single root node, and from there we can only traverse a tree.
 *  (b). For implementation, we considered supporting a
 *       `std::vector<std::shared_ptr> nodeUsers()` (in Node or another Graph
 *       class), but smart pointer doesn't play well here -- if we store the
 *       back-edge with `std::shared_ptr`, the node won't get automatically
 *       released when out-of-graph ref-count to the node becomes 0,
 *       because the used-node/graph still has a ref-count to the "should be
 *       deleted" user.
 *       One idea is to use `weak_ptr` and lazily prune these edges whenever
 *       `userNodes()` is called. But that feels slow, especially if it's called
 *       frequently during analysis. Need to investigate more.
 */
class Node {
  const std::vector<std::shared_ptr<Node>> inputs_;
  const Shape shape_;
  unsigned useCount_{0};

  // present if this node has been evaluated
  std::optional<Tensor> result_{std::nullopt};

  void nodeImplTypeCheck(NodeType expect, NodeType actual) const;

 protected:
  Node(std::vector<std::shared_ptr<Node>>&& inputs, Shape&& shape);

  // Limiting input get/set API to help enforce internal consistency.
  std::shared_ptr<Node> getInput(unsigned idx) const;

 public:
  virtual ~Node();

  // Metadata
  const std::vector<std::shared_ptr<Node>>& inputs() const;
  const Shape& shape() const;

  // immutable input update
  virtual std::shared_ptr<Node> mapInputs(
      std::function<std::shared_ptr<Node>(std::shared_ptr<Node>)>&& func) = 0;

  // Help lazy eval.
  const std::optional<Tensor>& getResult() const;
  void setResult(Tensor&& tensor);

  // Help decide when node fusion is beneficial; a comprehensive heuristics may
  // require `userNodes()`, but KISS for now.
  //
  // NOTE we differentiate uses for nodes here -- currently no requirement on
  // whether `Tensor::shallowCopy()` or indexing increments use count.
  unsigned getUseCount() const;
  // use carefully -- uses should be either Node or Tensor.
  void incUseCount();
  void decUseCount();

  // Convenient type checks
  bool isBinary() const;
  bool isCustom() const;
  bool isIndex() const;
  bool isScalar() const;
  bool isValue() const;
  bool isIndexedMerge() const;

  // Fast & safe casts
  virtual NodeType type() const = 0;

  template <typename T>
  T& impl() {
    nodeImplTypeCheck(T::nodeType, this->type());
    return *static_cast<T*>(this);
  }

  template <typename T>
  const T& impl() const {
    nodeImplTypeCheck(T::nodeType, this->type());
    return *static_cast<const T*>(this);
  }
};


/**
 * A trait for some generic Node functionalities.
 *
 * REQUIRED definition in derived class:
 *   public: static constexpr NodeType nodeType;
 */
template <typename Derived>
class NodeTrait : public Node {
 public:
  NodeTrait(std::vector<std::shared_ptr<Node>>&& inputs, Shape&& shape)
    : Node(std::move(inputs), std::move(shape)) {}

  NodeType type() const override {
    return Derived::nodeType;
  }
};

} // namespace fl
