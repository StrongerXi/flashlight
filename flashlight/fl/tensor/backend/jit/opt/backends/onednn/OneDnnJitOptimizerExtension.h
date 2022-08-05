/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/backend/jit/opt/JitOptimizerExtension.h"

namespace fl {

/**
 * JIT graph optimziation specific to OneDNN backend, e.g., node fusion to
 * leverage OneDNN post-ops.
 */
class OneDnnJitOptimizerExtension : public JitOptimizerExtension {
 public:
  static bool registered;

  std::shared_ptr<Node> optimize(std::shared_ptr<Node> node) override;
  bool isDataTypeSupported(const fl::dtype& dtype) const override;
};

} // namespace fl
