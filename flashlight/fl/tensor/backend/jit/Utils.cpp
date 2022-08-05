/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/backend/jit/Utils.h"

namespace fl {

bool isNodeUsedInGraphOnly(const std::shared_ptr<Node>& node) {
  // add 1 for the ref-count from `node`, which is assumed to be out-of-graph
  return node->numNodeUsers() + 1 == node.use_count();
}

} // namespace fl
