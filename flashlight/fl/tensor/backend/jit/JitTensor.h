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
 * A trait to turn an arbitrary Tensor type into a JitTensorBase.
 */
template <typename T>
class JitTensor : public JitTensorBase {
 protected:
  Tensor fromSharedNode(std::shared_ptr<SharedNode> sharedNode) const override {
    return toTensor<JitTensor>(sharedNode);
  }

  TensorBackend& wrappedBackend() const override {
    static TensorBackend& wrappedBackend = toTensor<T>().backend();
    return wrappedBackend;
  }

  Optimizer& optimizer() const override {
    static Optimizer optimizer(wrappedBackend());
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

  // allow use to create smart pointer of this derived class
  explicit JitTensor(std::shared_ptr<Node> node)
      : JitTensorBase(std::move(node)) {}
  explicit JitTensor(std::shared_ptr<SharedNode> sharedNode)
      : JitTensorBase(std::move(sharedNode)) {}

  JitTensor() : JitTensor({0}, fl::dtype::f32, nullptr, Location::Host) {}

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
      const Dim nRows,
      const Dim nCols,
      const Tensor& values,
      const Tensor& rowIdx,
      const Tensor& colIdx,
      StorageType storageType)
    : JitTensorBase(ValueNode::create(toTensor<T>(nRows, nCols, values, rowIdx, colIdx, storageType))) {}

  std::unique_ptr<TensorAdapterBase> clone() const override {
    // NOTE IR-captured computation semantics is immutable
    return std::make_unique<JitTensor>(node());
  }
};

} // namespace fl
