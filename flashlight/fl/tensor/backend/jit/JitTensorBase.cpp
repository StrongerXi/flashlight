/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/JitTensorBase.h"

#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"

#include <sstream>
#include <stdexcept>

#define FL_JIT_TENSOR_UNIMPLEMENTED \
  throw std::invalid_argument(       \
      "JitTensorBase::" + std::string(__func__) + " - unimplemented.");

namespace fl {

JitTensorBase::JitTensorBase(std::shared_ptr<Node> node)
  : JitTensorBase(std::make_shared<SharedData>(node)) {}

JitTensorBase::JitTensorBase(std::shared_ptr<SharedData> sharedData)
  : JitTensorBase(sharedData, std::make_shared<SharedIndexing>()) {}

JitTensorBase::JitTensorBase(
    std::shared_ptr<SharedData> sharedData,
    std::shared_ptr<SharedIndexing> sharedIndexing)
  : sharedData_(sharedData), sharedIndexing_(sharedIndexing) {}

JitTensorBase::~JitTensorBase() {}

void JitTensorBase::replaceNode(std::shared_ptr<Node> newNode) {
  if (hasIndexing()) {
    sharedIndexing_->replaceNode(newNode);
  } else {
    replaceDataNode(newNode);
  }
}

void JitTensorBase::replaceDataNode(std::shared_ptr<Node> newNode) {
  sharedData_->replaceNode(newNode);
}

const Tensor& JitTensorBase::getTensorOrEvalNode() {
  eval();
  return node()->getResult().value();
}

Tensor JitTensorBase::copy() {
  // TODO materialize to force copy? We need to agree on the semantics of
  // Tensor::copy()
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
// x = 42
// ......
// --->
// x' = 42
// ...... (x becomes x')
//
// TODO for simplicity, we fall back to Tensor assignment for all other
// assignment ops. Specialize when performance becomes an issue.
void JitTensorBase::assign(const Tensor& other) {
  if (hasIndexing()) {
    throw std::runtime_error(
        "[JitTensorBase::assign] Currently no support for indexed update");
  }
  replaceNode(toJitTensorBase(other).node());
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

FL_JIT_TENSOR_ASSIGN_OP(assign);                // =
FL_JIT_TENSOR_ASSIGN_BINOP(inPlaceAdd, +);      // +=
FL_JIT_TENSOR_ASSIGN_BINOP(inPlaceSubtract, -); // -=
FL_JIT_TENSOR_ASSIGN_BINOP(inPlaceMultiply, *); // *=
FL_JIT_TENSOR_ASSIGN_BINOP(inPlaceDivide, /);   // /=
#undef FL_JIT_TENSOR_ASSIGN_OP_SCALAR
#undef FL_JIT_TENSOR_ASSIGN_BINOP_TENSOR
#undef FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR
#undef FL_JIT_TENSOR_ASSIGN_OP
#undef FL_JIT_TENSOR_ASSIGN_BINOP

std::shared_ptr<Node> JitTensorBase::node() const {
  if (hasIndexing()) {
    return sharedIndexing_->getViewNode(sharedData_);
  } else {
    return sharedData_->node;
  }
}

void JitTensorBase::eval() {
  const auto node = this->node();
  if (!node->getResult().has_value()) {
    replaceNode(optimizer().optimize(node));
    // TODO consider updating `node` to a value node here, to help free up the
    // graph nodes. Investigate pros/cons of such graph-truncation
    evaluator().execute(this->node()); // node might have changed
  }
}

JitTensorBase& toJitTensorBase(const Tensor& tensor) {
  auto type = tensor.backendType();
  if (type != TensorBackendType::Jit) {
    std::ostringstream oss;
    oss << "[toJitTensor] expected JIT-backed tensor, got " << type;
    throw std::invalid_argument(oss.str());
  }
  return tensor.getAdapter<JitTensorBase>();
}

JitTensorBase& toJitTensorBase(Tensor& tensor) {
  return toJitTensorBase(static_cast<const Tensor&>(tensor));
}

} // namespace fl
