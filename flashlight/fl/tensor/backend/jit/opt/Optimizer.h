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
  // TODO cache?
public:
  Optimizer() = default;

  // Optimize the computation tree rooted at `node`
  std::shared_ptr<Node> optimize(std::shared_ptr<Node> node);
};

} // namespace fl
