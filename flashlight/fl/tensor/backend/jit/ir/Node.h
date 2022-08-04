/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/jit/ir/NodeType.h"

namespace fl {

/**
 * TODO a Node that represents Tensor ops.
 */
class Node {
  std::vector<std::shared_ptr<Node>> inputs_; // size should be fixed

  // Users within graph only.
  //
  // Considered using `weak_ptr` for possibly supporting
  // `std::vector<std::shared_ptr> nodeUsers()` in the future, but automatic
  // registration/deregistration doesn't play well in constructor/destructor.
  //
  // Maybe raw pointer is what we want in such cases, need to investigate more.
  unsigned numNodeUsers_{0};

  // present if this node has been evaluated
  std::optional<Tensor> result_{std::nullopt};

  // TODO
  void nodeImplTypeCheck(NodeType expect, NodeType actual) const;

 protected:
  // Limiting input get/set API to help enforce internal consistency.

  // TODO
  std::shared_ptr<Node> getInput(unsigned idx) const;

  /**
   * TODO
   */
  Node(std::vector<std::shared_ptr<Node>>&& inputs);

 public:
  /**
   * TODO
   */
  virtual ~Node();

  /**
   * TODO Return inputs nodes.
   */
  const std::vector<std::shared_ptr<Node>>& inputs() const;

  /**
   * TODO get evaluation result.
   * For JIT eval.
   */
  const std::optional<Tensor>& getResult() const;

  /**
   * TODO set evaluation result; throw if already set.
   * For JIT eval.
   */
  void setResult(Tensor&& tensor);

  /**
   * TODO
   * for matching temporary nodes (those w/o external users)
   */
  unsigned numNodeUsers() const;

  /**
   * TODO
   * for graph rewrite
   */
  void replaceInputNode(
      const std::shared_ptr<Node>& oldInput,
      const std::shared_ptr<Node>& newInput);

  /**
   * TODO
   */
  virtual NodeType type() const = 0;

  /**
   * TODO
   */
  template <typename T>
  T& impl() {
    nodeImplTypeCheck(T::nodeType, this->type());
    return *static_cast<T*>(this);
  }

  /**
   * TODO
   */
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
  /**
   * TODO
   */
  NodeTrait(std::vector<std::shared_ptr<Node>>&& inputs)
    : Node(std::move(inputs)) {}

  NodeType type() const override {
    return Derived::nodeType;
  }
};

} // namespace fl
