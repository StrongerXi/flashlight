/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorExtension.h"

#include "flashlight/fl/tensor/backend/jit/opt/backends/onednn/OneDnnJitOptimizerExtension.h"

namespace fl {

/****************** Jit Optimizer Extension Registration ******************/

FL_REGISTER_TENSOR_EXTENSION(
    OneDnnJitOptimizerExtension,
    TensorBackendType::OneDnn);

} // namespace fl
