/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/CustomNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/IndexNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/IndexedMergeNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"

namespace fl {

/**
 * A JIT graph evaluator; it dispatches to another Tensor Backend depending on
 * the JIT nodes.
 */
class Evaluator {
  // backend used for dispatching Tensor ops.
  TensorBackend& backend_;

  const Tensor evalBinaryNode(BinaryNode& node);
  const Tensor evalCustomNode(CustomNode& node);
  const Tensor evalIndexNode(IndexNode& node);
  const Tensor evalIndexedMergeNode(IndexedMergeNode& node);
  std::vector<Index> evalIndices(const std::vector<Index>& indices);
  const Tensor evalScalarNode(ScalarNode& node);
  const Tensor getTensorOrEvalNode(std::shared_ptr<Node> node);

public:
  // NOTE no assumption on `backend` -- it can even be another JIT.
  explicit Evaluator(TensorBackend& backend);

  // execute the entire computation tree rooted at `node`
  // 1. no op if result already set
  // 2. set result for all intermediate/final tensors evaluated
  // Not returning `Tensor` because we want to prevent copy.
  void execute(std::shared_ptr<Node> node);
};

} // namespace fl
