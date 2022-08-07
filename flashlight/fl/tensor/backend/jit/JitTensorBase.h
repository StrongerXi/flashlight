/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/backend/jit/JitBackend.h"
#include "flashlight/fl/tensor/backend/jit/eval/Evaluator.h"
#include "flashlight/fl/tensor/backend/jit/ir/Node.h"
#include "flashlight/fl/tensor/backend/jit/opt/Optimizer.h"

namespace fl {

/**
 * A JIT Tensor that wraps another backend -- it lazily builds up the
 * computation graph and delegates to the wrapped backend for execution.
 */
class JitTensorBase : public TensorAdapterBase {
 public:
  // shared among shallow copies
  // NOTE !!Internal use only!!
  // TODO made public to enable `std::make_shared`.
  // Consider using same trick as `...Node::create` to make this protected
  struct SharedNode {
    std::shared_ptr<Node> node;
    SharedNode(std::shared_ptr<Node> node) : node(node) {
      node->incUseCount(); // shallow copies counts as 1 use
    }
    ~SharedNode() {
      node->decUseCount();
    }
  };

 private:
  // for shallow copy
  std::shared_ptr<SharedNode> sharedNode_;

  // take care of inc/dec node use count
  void replaceNode(std::shared_ptr<Node> newNode);

  // return the wrapped tensor, not a JitTensorBase
  const Tensor& getTensorOrEvalNode();

 protected:
  // this allows us to create an instance of derived class
  virtual
  Tensor fromSharedNode(std::shared_ptr<SharedNode> sharedNode) const = 0;

  // let derived class manage the wrapped backend
  virtual TensorBackend& wrappedBackend() const = 0;

  // allow JitTensor<T> to potentially inject things into Optimizer/Evaluator
  virtual Optimizer& optimizer() const = 0;
  virtual Evaluator& evaluator() const = 0;

  // JitTensorBase manages the backend-agnostic JIT node.
  JitTensorBase(std::shared_ptr<Node> node);
  JitTensorBase(std::shared_ptr<SharedNode> sharedNode);

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

  // NOTE `const` w.r.t. the underlying Tensor this represents.
  std::shared_ptr<Node> node() const;
  void eval();

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
