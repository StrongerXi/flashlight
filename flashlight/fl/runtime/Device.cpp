/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/runtime/Device.h"

namespace fl {

  DeviceManager& DeviceManager::getInstance() {
    static DeviceManager instance;
    return instance;
  }

  const Device& DeviceManager::getActiveDevice(const DeviceType type) const {
    return *activeDevices_.at(type);
  }

  const Device& DeviceManager::setActiveDevice(
    const DeviceType type, const Device& device
  ) {
   // TODO Check device is tracked?
   auto oldActiveDevice = activeDevices_.at(type);
   // TODO compilation stupidity
   activeDevices_.insert(type, &device);
   return *oldActiveDevice;
  }
} // namespace fl
