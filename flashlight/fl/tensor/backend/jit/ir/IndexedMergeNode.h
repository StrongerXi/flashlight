/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

namespace fl {

/**
 * A node that represents an immutable indexed update.
 * let output = IndexedMergeNode(indexedNode, indices, mergeSourceNode), we have
 *   1. output.shape == indexedNode.shape
 *   2. values of output at indices = mergeSourceNode
 *   3. values of output outside indices = indexedNode
 */
class IndexedMergeNode : public NodeTrait<IndexedMergeNode> {
  const std::vector<Index> indices_;

  // helps indexing into inputs
  static constexpr unsigned indexedNodeIdx = 0;
  static constexpr unsigned mergeSourceNodeIdx = 1;

  // intentionally kept private to control allocation
  IndexedMergeNode(
      Node* indexedNode,
      const std::vector<Index>& indices,
      Node* mergeSourceNode);

 public:
  static constexpr NodeType nodeType = NodeType::IndexedMerge;

  static IndexedMergeNode* create(
      Node* indexedNode,
      const std::vector<Index>& indices,
      Node* mergeSourceNode);

  Node* indexedNode() const;
  const std::vector<Index>& indices() const;
  Node* mergeSourceNode() const;
};

} // namespace fl
