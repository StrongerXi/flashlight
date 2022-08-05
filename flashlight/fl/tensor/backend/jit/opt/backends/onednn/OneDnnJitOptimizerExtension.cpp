/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/opt/backends/onednn/OneDnnJitOptimizerExtension.h"

#include "flashlight/fl/tensor/backend/jit/opt/backends/onednn/BinaryOpFuser.h"
#include "flashlight/fl/tensor/backend/onednn/OneDnnBackend.h"

namespace fl {

std::shared_ptr<Node> OneDnnJitOptimizerExtension::optimize(
    std::shared_ptr<Node> node) {
  node = BinaryOpFuser::apply(node);
  return node;
}

bool OneDnnJitOptimizerExtension::isDataTypeSupported(
    const fl::dtype& dtype) const {
  return OneDnnBackend::getInstance().isDataTypeSupported(dtype);
}

} // namespace fl
