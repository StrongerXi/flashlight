/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/IndexedMergeNode.h"

#include <stdexcept>

namespace fl {

std::shared_ptr<IndexedMergeNode> IndexedMergeNode::create(
    std::shared_ptr<Node> indexedNode,
    std::vector<Index> indices,
    std::shared_ptr<Node> mergeSourceNode) {
  return std::make_shared<IndexedMergeNode>(
      PrivateHelper{},
      indexedNode,
      std::make_shared<const std::vector<Index>>(std::move(indices)),
      mergeSourceNode);
}

IndexedMergeNode::IndexedMergeNode(
    const PrivateHelper&,
    std::shared_ptr<Node> indexedNode,
    std::shared_ptr<const std::vector<Index>> indices,
    std::shared_ptr<Node> mergeSourceNode) :
  NodeTrait({ indexedNode, mergeSourceNode }, Shape(indexedNode->shape())),
  indices_(indices) {}

std::shared_ptr<Node> IndexedMergeNode::indexedNode() const {
  return getInput(indexedNodeIdx);
}

const std::vector<Index>& IndexedMergeNode::indices() const {
  return *indices_;
}

std::shared_ptr<Node> IndexedMergeNode::mergeSourceNode() const {
  return getInput(mergeSourceNodeIdx);
}

std::shared_ptr<Node> IndexedMergeNode::mapInputs(
    std::function<std::shared_ptr<Node>(std::shared_ptr<Node>)>&& func) {
  // TODO map tensor in indices as well
  return std::make_shared<IndexedMergeNode>(
      PrivateHelper{}, func(indexedNode()), indices_, func(mergeSourceNode()));
}

} // namespace fl
