/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorExtension.h"
#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

namespace fl {

/**
 * Tensor Extension to enable backend-specific JIT graph optimization.
 */
class JitOptimizerExtension : public TensorExtension<JitOptimizerExtension> {
 public:
  static constexpr auto extensionType = TensorExtensionType::JitOptimizer;

  /**
   * Optimize the computation tree rooted at `node`.
   */
  virtual std::shared_ptr<Node> optimize(std::shared_ptr<Node> node) = 0;
};

} // namespace fl
