/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

namespace fl {

/**
 * Levearge OneDNN's binary post-ops to (1) avoid redundant memory allocation
 * and (2) reduce # of primitive launches.
 *
 * NOTE due to OneDNN limitation, post-op can only apply to rhs argument.
 *
 * n1   n2
 *  \  /
 *   b2   n3
 *    \  /
 *     b1
 *
 * --> (Assume only b1 needs to be materialized)
 *
 *    n1 n2 n3
 *     \ |  /
 * ---------------- CustomNode
 * | n1 = n1 + n2 |
 * | n1 = n2 + n3 |
 * ----------------
 *
 * Algorithm:
 * A typical 2-pass optimization, where the first pass collects information to
 * guide the second pass on what's profitable to fuse.
 *
 * TODO post-op fusion can be a lot more general than this, doesn't need to be a
 * chain of binary ops only.
 */
class BinaryOpFuser {
  struct BinopInfo;

  // NOTE Safe to use raw ptr for efficiency here since we never dereference.

  // Max # of consecutive nodes that can be profitably fused together along a
  // valid (lhs only unless op is commutative) input chain.
  const std::unordered_map<Node*, unsigned> profitableFuseChainLengths_;

  // Avoid re-visit, since fuser only need to apply once to each node.
  std::unordered_set<Node*> visited_{};

  // private ctor to set up state.
  BinaryOpFuser(std::shared_ptr<Node> root);

  // 1. Fuse _along_ some path from `node`.
  // 2. recursively optimize other inputs along the fused path.
  std::shared_ptr<Node> rewriteFrom(std::shared_ptr<Node> node);

  // Same as the other one, with more context.
  std::shared_ptr<Node> rewriteFrom(
      std::shared_ptr<Node> node,
      std::vector<BinopInfo>& accumulatedBinops);

  // Actual fusion of a path
  std::shared_ptr<Node> fuseAccumulatedBinops(
      std::shared_ptr<Node> node,
      std::vector<BinopInfo>& accumualtedBinops);

 public:
  static std::shared_ptr<Node> apply(std::shared_ptr<Node> root);
};

} // namespace fl
