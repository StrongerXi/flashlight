/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/IndexNode.h"

#include <stdexcept>

namespace fl {

namespace {

Dim ceilDiv(Dim numerator, Dim denominator) {
  return (numerator + denominator - 1) / denominator;
}

Shape inferIndexedShape(const Shape& shape, const std::vector<Index>& indices) {
  bool encounteredTensorIdx = false;
  std::vector<Dim> indexedDims;
  // TODO check indexing & shape validity
  for (unsigned i = 0; i < indices.size(); i++) {
    const auto& idx = indices[i];
    switch (idx.type()) {
      case detail::IndexType::Tensor: {
        if (encounteredTensorIdx) {
          throw std::runtime_error(
              "[inferIndexedShape]: No support for multiple Tensor index");
        }
        encounteredTensorIdx = true;
        const auto& tensorIdx = idx.get<Tensor>();
        const auto& tensorIdxDims = tensorIdx.shape().get();
        indexedDims.insert(
            indexedDims.end(), tensorIdxDims.begin(), tensorIdxDims.end());
        break;
      }
      case detail::IndexType::Span: {
        indexedDims.push_back(shape[i]);
        break;
      }
      case detail::IndexType::Range: {
        // TODO refactor with common index canonicalization logic in OneDnnTensor
        const auto& rangeIdx = idx.get<range>();
        const auto start = rangeIdx.start();
        const auto end = rangeIdx.end().value_or(shape.dim(i));
        const auto stride = rangeIdx.stride();
        if (start < 0 || end < 0 || stride <= 0) {
          throw std::runtime_error(
              "[inferIndexedShape]: Unsupported range index values");
        }
        indexedDims.push_back(ceilDiv(end - start, stride));
        break;
      }
      case detail::IndexType::Literal:
        continue; // dimension is reduced
      default:
        throw std::invalid_argument("[inferIndexedShape]: unknown index type.");
    }
  }
  return Shape(indexedDims);
}

} // namespace

IndexNode* IndexNode::create(Node* indexedNode, const std::vector<Index>& indices) {
  return new IndexNode(indexedNode, indices);
}

IndexNode::IndexNode(Node* indexedNode, const std::vector<Index>& indices) :
  NodeTrait({ indexedNode }, inferIndexedShape(indexedNode->shape(), indices)),
  indices_(indices) {}

Node* IndexNode::indexedNode() const {
  return getInput(indexedNodeIdx);
}

const std::vector<Index>& IndexNode::indices() const {
  return indices_;
}

} // namespace fl
