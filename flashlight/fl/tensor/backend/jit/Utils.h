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
 * TODO
 *
 * NOTE A conservative check; please
 * 1. ASSUME `node` is a reference to out-of-graph shared_ptr to node.
 * 2. minimize shared_ptr ref-count before using this.
 */
bool isNodeUsedInGraphOnly(const std::shared_ptr<Node>& node);

} // namespace fl
