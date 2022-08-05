/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

namespace fl {

/**
 * A JIT tree optimizer.
 */
class Optimizer {
  // backend used for optional JIT optimizer extension
  TensorBackend& backend_;

public:
  explicit Optimizer(TensorBackend& backend);

  // Optimize the computation tree rooted at `node`
  std::shared_ptr<Node> optimize(std::shared_ptr<Node> node);
};

} // namespace fl
