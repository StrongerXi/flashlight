/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/IndexedMergeNode.h"

#include <stdexcept>

namespace fl {

IndexedMergeNode::IndexedMergeNode(
    Node* indexedNode,
    const std::vector<Index>& indices,
    Node* mergeSourceNode) :
  NodeTrait({ indexedNode, mergeSourceNode }, Shape(indexedNode->shape())),
  indices_(indices) {}

IndexedMergeNode* IndexedMergeNode::create(
    Node* indexedNode,
    const std::vector<Index>& indices,
    Node* mergeSourceNode) {
  return new IndexedMergeNode(indexedNode, indices, mergeSourceNode);
}

Node* IndexedMergeNode::indexedNode() const {
  return getInput(indexedNodeIdx);
}

const std::vector<Index>& IndexedMergeNode::indices() const {
  return indices_;
}

Node* IndexedMergeNode::mergeSourceNode() const {
  return getInput(mergeSourceNodeIdx);
}

} // namespace fl
