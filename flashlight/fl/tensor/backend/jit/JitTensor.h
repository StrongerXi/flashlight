/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/backend/jit/JitTensorBase.h"
#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"

namespace fl {

/**
 * TODO
 * A trait to turn an arbitrary Tensor type into a tensor with JIT.
 */
template <typename T>
class JitTensor : public JitTensorBase {
 protected:
  std::unique_ptr<TensorAdapterBase> fromNode(
      std::shared_ptr<Node> node) const override {
    return std::make_unique<JitTensor>(node);
  }

  TensorBackend& wrappedBackend() const override {
    static TensorBackend& wrappedBackend = toTensor<T>().backend();
    return wrappedBackend;
  }

  Optimizer& optimizer() const override {
    static Optimizer optimizer;
    return optimizer;
  }

  Evaluator& evaluator() const override {
    static Evaluator evaluator(wrappedBackend());
    return evaluator;
  }

 public:
  // 1 static instance per jitted T.
  JitBackend& backend() const override {
    auto creator = [](std::shared_ptr<Node> node) {
      return toTensor<JitTensor>(node);
    };
    static JitBackend backend(wrappedBackend(), creator);
    return backend;
  }

  explicit JitTensor(std::shared_ptr<Node> node)
      : JitTensorBase(std::move(node)) {}

  /**
   * Construct a JitTensorBase using some data.
   *
   * @param[in] shape the shape of the new tensor
   * @param[in] ptr the buffer containing underlying tensor data
   * @param[in] type the type of the new tensor
   * @param[in] memoryLocation the location of the buffer
   */
  JitTensor(
      const Shape& shape,
      fl::dtype type,
      const void* ptr,
      Location memoryLocation)
  : JitTensorBase(ValueNode::create(toTensor<T>(shape, type, ptr, memoryLocation))) {}

  // Constructor for a sparse JitTensorBase. Can throw if unimplemented.
  JitTensor(
      const Dim /* nRows */,
      const Dim /* nCols */,
      const Tensor& /* values */,
      const Tensor& /* rowIdx */,
      const Tensor& /* colIdx */,
      StorageType /* storageType */)
      // TODO garbage, merely to allow constructor to compile
      : JitTensorBase(nullptr) {
    throw std::runtime_error("[JitTensor] Regular constructor not supported");
  }

  std::unique_ptr<TensorAdapterBase> clone() const override {
    // TODO copy node
    throw std::runtime_error("[JitTensor] clone not supported");
  }
};

} // namespace fl
