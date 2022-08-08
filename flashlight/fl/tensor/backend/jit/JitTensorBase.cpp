/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/JitTensorBase.h"

#include "flashlight/fl/tensor/backend/jit/ir/IndexedMergeNode.h"

#include <sstream>
#include <stdexcept>

#define FL_JIT_TENSOR_UNIMPLEMENTED \
  throw std::invalid_argument(      \
      "JitTensorBase::" + std::string(__func__) + " - unimplemented.");

namespace fl {

struct JitTensorBase::SharedData {
  Node* node;

  SharedData(Node* node) : node(node) {
    node->incRefCount(); // shallow copies counts as 1 use
  }

  ~SharedData() {
    node->decRefCount();
  }

  void replaceNode(Node* newNode) {
    newNode->incRefCount();
    node->decRefCount();
    node = newNode;
  }
};

// `t(x)` and `t(x)(y)` share the same node, but different index; but we can't
// materialize into view node because the shared node may get updated due to
// assignment, so we lazily materialize index into view node.
struct JitTensorBase::SharedIndexing {
  // nullptr & std::nullopt if we haven't materialized any indexing
  Node* dataNode{nullptr};
  std::optional<Node*> viewNode{std::nullopt};
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
      viewNode.value()->decRefCount();
    }
  }

  void replaceNode(Node* newNode) {
    // TODO should we enforce `indexNode.has_value()` here?
    newNode->incRefCount();
    if (viewNode.has_value()) {
      viewNode.value()->decRefCount();
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

  Node* getViewNode(std::shared_ptr<SharedData> sharedData) {
    if (dataNode != sharedData->node) {
      // must materialize
      const auto newIndexNode =
        IndexNode::create(sharedData->node, getIndex());
      newIndexNode->incRefCount();
      if (viewNode.has_value()) {
        viewNode.value()->decRefCount();
      }
      viewNode = newIndexNode;
      dataNode = sharedData->node;
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

JitTensorBase::JitTensorBase(Node* node)
    : JitTensorBase(std::make_shared<SharedData>(node)) {}

JitTensorBase::JitTensorBase(std::shared_ptr<SharedData> sharedData)
    : JitTensorBase(sharedData, std::make_shared<SharedIndexing>()) {}

JitTensorBase::JitTensorBase(
    std::shared_ptr<SharedData> sharedData,
    std::shared_ptr<SharedIndexing> sharedIndexing)
    : sharedData_(sharedData), sharedIndexing_(sharedIndexing) {}

JitTensorBase::~JitTensorBase() {}

void JitTensorBase::replaceNode(Node* newNode) const {
  if (hasIndexing()) {
    sharedIndexing_->replaceNode(newNode);
  } else {
    replaceDataNode(newNode);
  }
}

bool JitTensorBase::hasIndexing() const {
  return !sharedIndexing_->indexings.empty();
}

void JitTensorBase::replaceDataNode(Node* newNode) const {
  sharedData_->replaceNode(newNode);
}

const Tensor& JitTensorBase::getTensorOrEvalNode() const {
  if (!node()->getResult().has_value()) {
    eval();
  }
  return node()->getResult().value();
}

Tensor JitTensorBase::copy() {
  // Since a node's computation result is immutable, copy is free.
  return Tensor(clone());
}

Tensor JitTensorBase::shallowCopy() {
  // NOTE IR-captured computation semantics is immutable
  return fromSharedData(sharedData_, sharedIndexing_);
}

TensorBackendType JitTensorBase::backendType() const {
  return TensorBackendType::Jit;
}

const Shape& JitTensorBase::shape() {
  return node()->shape();
}

fl::dtype JitTensorBase::type() {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

bool JitTensorBase::isSparse() {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

Location JitTensorBase::location() {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

void JitTensorBase::scalar(void* /* out */) {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

void JitTensorBase::device(void** /* out */) {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

void JitTensorBase::host(void* /* out */) {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

void JitTensorBase::unlock() {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

bool JitTensorBase::isLocked() {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

bool JitTensorBase::isContiguous() {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

Shape JitTensorBase::strides() {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

const Stream& JitTensorBase::stream() const {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

Tensor JitTensorBase::astype(const dtype /* type */) {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

Tensor JitTensorBase::index(const std::vector<Index>& indices) {
  return fromSharedData(sharedData_, sharedIndexing_->applyIndices(indices));
}

Tensor JitTensorBase::flatten() const {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

Tensor JitTensorBase::flat(const Index& /* idx */) const {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

Tensor JitTensorBase::asContiguousTensor() {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

void JitTensorBase::setContext(void* /* context */) {
  // Used to store arbitrary data on a Tensor - can be a noop.
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

void* JitTensorBase::getContext() {
  // Used to store arbitrary data on a Tensor - can be a noop.
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

std::string JitTensorBase::toString() {
  return getTensorOrEvalNode().toString();
}

std::ostream& JitTensorBase::operator<<(std::ostream& /* ostr */) {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

/******************** Assignment Operators ********************/
// NOTE Think SSA:
// x += 42
// ......
// --->
// x' = x + 42
// ...... (uses of x becomes x')
void JitTensorBase::assign(const Tensor& other) {
  if (hasIndexing()) {
    // Again, SSA, but this time we can't fall back to existing ops, i.e., we
    // must increase the expressivity of our IR, thus a new IR node --
    // `IndexedMerge`. NOTE We let backend can optimize such patterns to avoid
    // redundant memory alloc, e.g., by dispatching to the more efficient
    // `Tensor::[...]assign(...)`
    const auto& indices = sharedIndexing_->getIndex();
    const auto thisDataNode = this->sharedData_->node;
    const auto otherNode = toJitTensorBase(other).node();
    const auto mergeNode =
        IndexedMergeNode::create(thisDataNode, indices, otherNode);
    replaceDataNode(mergeNode);
  } else {
    replaceNode(toJitTensorBase(other).node());
  }
}

#define FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, TYPE)                \
  void JitTensorBase::OP(const TYPE& scalar) {                  \
    const auto dtype = dtype_traits<TYPE>::ctype;               \
    this->assign(backend().full(this->shape(), scalar, dtype)); \
  }

#define FL_JIT_TENSOR_ASSIGN_BINOP_TENSOR(OP, BINOP) \
  void JitTensorBase::OP(const Tensor& other) {      \
    this->assign(this->shallowCopy() BINOP other);   \
  }

#define FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, TYPE) \
  void JitTensorBase::OP(const TYPE& scalar) {             \
    this->assign(this->shallowCopy() BINOP scalar);        \
  }

#define FL_JIT_TENSOR_ASSIGN_OP(OP)                   \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, double);         \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, float);          \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, int);            \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, unsigned);       \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, bool);           \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, char);           \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, unsigned char);  \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, short);          \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, unsigned short); \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, long);           \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, unsigned long);  \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, long long);      \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, unsigned long long);

#define FL_JIT_TENSOR_ASSIGN_BINOP(OP, BINOP)                   \
  FL_JIT_TENSOR_ASSIGN_BINOP_TENSOR(OP, BINOP);                 \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, double);         \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, float);          \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, int);            \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, unsigned);       \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, bool);           \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, char);           \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, unsigned char);  \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, short);          \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, unsigned short); \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, long);           \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, unsigned long);  \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, long long);      \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, unsigned long long);

FL_JIT_TENSOR_ASSIGN_OP(assign); // =
FL_JIT_TENSOR_ASSIGN_BINOP(inPlaceAdd, +); // +=
FL_JIT_TENSOR_ASSIGN_BINOP(inPlaceSubtract, -); // -=
FL_JIT_TENSOR_ASSIGN_BINOP(inPlaceMultiply, *); // *=
FL_JIT_TENSOR_ASSIGN_BINOP(inPlaceDivide, /); // /=
#undef FL_JIT_TENSOR_ASSIGN_OP_SCALAR
#undef FL_JIT_TENSOR_ASSIGN_BINOP_TENSOR
#undef FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR
#undef FL_JIT_TENSOR_ASSIGN_OP
#undef FL_JIT_TENSOR_ASSIGN_BINOP

Node* JitTensorBase::node() const {
  if (hasIndexing()) {
    return sharedIndexing_->getViewNode(sharedData_);
  } else {
    return sharedData_->node;
  }
}

void JitTensorBase::eval() const {
  if (!node()->getResult().has_value()) {
    replaceNode(optimizer().optimize(node()));
    evaluator().eval(node());
  }
}

const JitTensorBase& toJitTensorBase(const Tensor& tensor) {
  return toJitTensorBase(const_cast<Tensor&>(tensor));
}

JitTensorBase& toJitTensorBase(Tensor& tensor) {
  auto type = tensor.backendType();
  if (type != TensorBackendType::Jit) {
    std::ostringstream oss;
    oss << "[toJitTensorBase] expected JIT-backed tensor, got " << type;
    throw std::invalid_argument(oss.str());
  }
  return tensor.getAdapter<JitTensorBase>();
}

} // namespace fl
