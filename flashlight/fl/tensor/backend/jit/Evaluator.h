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
#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"

namespace fl {

/**
 * TODO
 * JIT graph evaluator.
 */
class Evaluator {
  // backend used for carrying out Tensor computation
  TensorBackend& backend_;

  Tensor evalBinaryNode(BinaryNode& node);
  Tensor evalCustomNode(CustomNode& node);
  Tensor evalScalarNode(ScalarNode& node);
  Tensor evalNode(std::shared_ptr<Node> node);
  const Tensor& getTensorOrEvalNode(std::shared_ptr<Node> node);

public:
  explicit Evaluator(TensorBackend& backend);

  // execute the entire computation tree rooted at `node`
  // 1. no op if result already set
  // 2. set result for all intermediate/final tensors evaluated
  Tensor execute(std::shared_ptr<Node> node);
};

} // namespace fl
