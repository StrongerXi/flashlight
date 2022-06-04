/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>

namespace fl {

/**
 * TODO
 */
enum class DeviceType {
  X64,
  CUDA,
  AMD,
  IPU
};

/**
 * TODO
 */
class Device {
  // TODO
};

/**
 * TODO
 */
class DeviceManager {
  // TODO how are these populated/initialized?
  std::vector<std::unique_ptr<Device>> devices_;
  // device -> FL ID (TODO auto-incremented, wrapped runtime ID?)
  // TODO why?
  // (Type, Correct runtime id) -> Device
  std::unordered_map<DeviceType, std::unordered_map<int, Device const*>> ids_;
  std::unordered_map<DeviceType, Device const*> activeDevices_;

  DeviceManager() = default;
  DeviceManager(DeviceManager&) = delete;
  DeviceManager(DeviceManager&&) = delete;

 public:
  static DeviceManager& getInstance();
  // TODO test
  const Device& getActiveDevice(const DeviceType type) const;
  const Device& setActiveDevice(const DeviceType type, const Device& device);
};
} // namespace fl
