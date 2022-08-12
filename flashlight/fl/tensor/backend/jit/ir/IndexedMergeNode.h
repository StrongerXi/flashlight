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
class IndexedMergeNode :
  public NodeTrait<IndexedMergeNode>,
  public std::enable_shared_from_this<IndexedMergeNode> {
  const std::shared_ptr<const std::vector<Index>> indices_;

  // a trick to enable `std::make_shared` with effecitvely private constructor
  struct PrivateHelper{};

  // helps indexing into inputs
  static constexpr unsigned indexedNodeIdx = 0;
  static constexpr unsigned mergeSourceNodeIdx = 1;

 public:
  static constexpr NodeType nodeType = NodeType::IndexedMerge;

  static std::shared_ptr<IndexedMergeNode> create(
      std::shared_ptr<Node> indexedNode,
      std::vector<Index> indices,
      std::shared_ptr<Node> mergeSourceNode);

  std::shared_ptr<Node> indexedNode() const;
  const std::vector<Index>& indices() const;
  std::shared_ptr<Node> mergeSourceNode() const;

  std::shared_ptr<Node> mapInputs(
      std::function<std::shared_ptr<Node>(std::shared_ptr<Node>)>&& func
  ) override;

  // intentionally kept unusable publicly
  IndexedMergeNode(
      const PrivateHelper&,
      std::shared_ptr<Node> indexedNode,
      std::shared_ptr<const std::vector<Index>> indices,
      std::shared_ptr<Node> mergeSourceNode);
};

} // namespace fl
