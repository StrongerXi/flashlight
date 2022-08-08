/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/backend/jit/JitBackend.h"
#include "flashlight/fl/tensor/backend/jit/eval/Evaluator.h"
#include "flashlight/fl/tensor/backend/jit/ir/Node.h"
#include "flashlight/fl/tensor/backend/jit/ir/IndexNode.h"
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
  struct SharedData {
    std::shared_ptr<Node> node;

    SharedData(std::shared_ptr<Node> node) : node(node) {
      node->incUseCount(); // shallow copies counts as 1 use
    }

    ~SharedData() {
      node->decUseCount();
    }

    void replaceNode(std::shared_ptr<Node> newNode) {
      newNode->incUseCount();
      node->decUseCount();
      node = newNode;
    }
  };

  // `t(x)` and `t(x)(y)` share the same node, but different index; but we can't
  // materialize into view node because the shared node may get updated due to
  // assignment, so we lazily materialize index into view node.
  struct SharedIndexing {
    // nullptr & std::nullopt if we haven't materialized any indexing
    Node* dataNode{nullptr};
    std::optional<std::shared_ptr<Node>> viewNode{std::nullopt};
    // TODO consider
    // 1. using an immutable linked-list here to speed things up
    // 2. making `std::vector<Index>` into an immutable class and use it as
    //    shared_ptr, since it's all readonly -- JitTensor gives us free
    //    immutability for Tensor index.
    // 3. index merging (might require canonicalization) -- maybe as an
    //    optimization pass, need to think more.
    const std::vector<std::vector<Index>> indexings{};

    SharedIndexing() = default;

    SharedIndexing(std::vector<std::vector<Index>> indexings)
      : indexings(std::move(indexings)) {}

    ~SharedIndexing() {
      if (viewNode.has_value()) {
        viewNode.value()->decUseCount();
      }
    }

    void replaceNode(std::shared_ptr<Node> newNode) {
      // TODO should we enforce `indexNode.has_value()` here?
      newNode->incUseCount();
      if (viewNode.has_value()) {
        viewNode.value()->decUseCount();
      }
      viewNode = newNode;
    }

    const std::vector<Index>& getIndex() {
      if (indexings.size() > 1) {
        // TODO rare, but see comment around `indexings` field
        throw std::runtime_error(
            "[SharedIndexing::getIndex] Currently no support for nested indexing");
      }
      return indexings[0];
    }

    std::shared_ptr<Node> getViewNode(
        std::shared_ptr<SharedData> sharedData) {
      const auto rawDataNodePtr = sharedData->node.get();
      if (dataNode != rawDataNodePtr) {
        // must materialize
        const auto newIndexNode =
          IndexNode::create(sharedData->node, getIndex());
        newIndexNode->incUseCount();
        if (viewNode.has_value()) {
          viewNode.value()->decUseCount();
        }
        viewNode = newIndexNode;
        dataNode = rawDataNodePtr;
      }
      return viewNode.value();
    }

    std::shared_ptr<SharedIndexing> applyIndices(std::vector<Index> indices) {
      std::vector<std::vector<Index>> newIndexings = this->indexings;
      newIndexings.push_back(std::move(indices));
      // refreshes indexNode cache,
      // TODO ideally we coulve retain some info here for performance, e.g., the
      // next IndexNode can build on top of existing one (if any).
      return std::make_shared<SharedIndexing>(newIndexings);
    }
  };

 private:
  // for shallow copy
  std::shared_ptr<SharedData> sharedData_;
  std::shared_ptr<SharedIndexing> sharedIndexing_;
  // TODO make sharedIndexing_ optional since most tensors likely aren't
  // indexed. It makes the design cleaner since we can have the invariant --
  // `!indexings.empty()`

  bool hasIndexing() const {
    return !sharedIndexing_->indexings.empty();
  }

  // take care of inc/dec node use count; also see `node()` documentation for
  // interpretation of `Node` here.
  void replaceNode(std::shared_ptr<Node> newNode);
  void replaceDataNode(std::shared_ptr<Node> newNode);

  // return the wrapped tensor, not a JitTensorBase
  // `const` w.r.t. the underlying Tensor this represents.
  const Tensor& getTensorOrEvalNode();

  Tensor fromNode(std::shared_ptr<Node> node) const;

 protected:
  // this allows us to create an instance of derived class
  virtual
  Tensor fromSharedData(
      std::shared_ptr<SharedData> sharedData,
      std::shared_ptr<SharedIndexing> sharedIndexing) const = 0;

  // let derived class manage the wrapped backend
  virtual TensorBackend& wrappedBackend() const = 0;

  // allow JitTensor<T> to potentially inject things into Optimizer/Evaluator
  virtual Optimizer& optimizer() const = 0;
  virtual Evaluator& evaluator() const = 0;

  // JitTensorBase manages the backend-agnostic JIT node.
  JitTensorBase(std::shared_ptr<Node> node);
  JitTensorBase(std::shared_ptr<SharedData> sharedData);
  JitTensorBase(
      std::shared_ptr<SharedData> sharedData,
      std::shared_ptr<SharedIndexing> sharedIndexing);

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

  // NOTE
  // 1. `const` w.r.t. the underlying Tensor this represents.
  // 2. return a node that represents the underlying Tensor, i.e., if there is
  //    indexing, take care of it.
  // TODO more specific name -- viewNode?
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
