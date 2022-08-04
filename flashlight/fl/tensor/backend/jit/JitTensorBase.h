/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/backend/jit/JitBackend.h"
#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

namespace fl {

/**
 * A JIT Tensor that wraps another backend -- it lazily builds up the
 * computation graph and delegates to the wrapped backend for execution.
 */
class JitTensorBase : public TensorAdapterBase {
  std::shared_ptr<Node> node_;

  // take care of inc/dec node use count
  void replaceNode(std::shared_ptr<Node> newNode);

 protected:
  // this allows us to create an instance of derived class
  virtual Tensor fromNode(std::shared_ptr<Node> node) const = 0;

  // let derived class manage the wrapped backend
  virtual TensorBackend& wrappedBackend() const = 0;

  // JitTensorBase manages the backend-agnostic JIT node.
  JitTensorBase(std::shared_ptr<Node> node);

 public:
  virtual ~JitTensorBase() override;
  TensorBackendType backendType() const override;
  virtual JitBackend& backend() const override = 0;
  Tensor copy() override;
  Tensor shallowCopy() override;
  const Shape& shape() override;
  dtype type() override;
  bool isSparse() override;
  Location location() override;
  void scalar(void* out) override;
  void device(void** out) override;
  void host(void* out) override;
  void unlock() override;
  bool isLocked() override;
  bool isContiguous() override;
  Shape strides() override;
  const Stream& stream() const override;
  Tensor astype(const dtype type) override;
  Tensor index(const std::vector<Index>& indices) override;
  Tensor flatten() const override;
  Tensor flat(const Index& idx) const override;
  Tensor asContiguousTensor() override;
  void setContext(void* context) override;
  void* getContext() override;
  std::string toString() override;
  std::ostream& operator<<(std::ostream& ostr) override;

  std::shared_ptr<Node> node();

  /******************** Assignment Operators ********************/
#define ASSIGN_OP_TYPE_STUB(OP, TYPE) void OP(const TYPE& val) override;

#define ASSIGN_OP_STUB(OP)                 \
  ASSIGN_OP_TYPE_STUB(OP, Tensor);         \
  ASSIGN_OP_TYPE_STUB(OP, double);         \
  ASSIGN_OP_TYPE_STUB(OP, float);          \
  ASSIGN_OP_TYPE_STUB(OP, int);            \
  ASSIGN_OP_TYPE_STUB(OP, unsigned);       \
  ASSIGN_OP_TYPE_STUB(OP, bool);           \
  ASSIGN_OP_TYPE_STUB(OP, char);           \
  ASSIGN_OP_TYPE_STUB(OP, unsigned char);  \
  ASSIGN_OP_TYPE_STUB(OP, short);          \
  ASSIGN_OP_TYPE_STUB(OP, unsigned short); \
  ASSIGN_OP_TYPE_STUB(OP, long);           \
  ASSIGN_OP_TYPE_STUB(OP, unsigned long);  \
  ASSIGN_OP_TYPE_STUB(OP, long long);      \
  ASSIGN_OP_TYPE_STUB(OP, unsigned long long);

  ASSIGN_OP_STUB(assign); // =
  ASSIGN_OP_STUB(inPlaceAdd); // +=
  ASSIGN_OP_STUB(inPlaceSubtract); // -=
  ASSIGN_OP_STUB(inPlaceMultiply); // *=
  ASSIGN_OP_STUB(inPlaceDivide); // /=
#undef ASSIGN_OP_TYPE
#undef ASSIGN_OP
};

JitTensorBase& toJitTensorBase(const Tensor& tensor);
JitTensorBase& toJitTensorBase(Tensor& tensor);

} // namespace fl