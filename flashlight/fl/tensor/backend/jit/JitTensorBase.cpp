/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/JitTensorBase.h"

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/IndexedMergeNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"

#include <memory>
#include <sstream>
#include <stdexcept>

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

Tensor JitTensorBase::fromNode(std::shared_ptr<Node> node) const {
  return fromSharedData(
      std::make_shared<SharedData>(node),
      std::make_shared<SharedIndexing>());
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
  // TODO
  // Consider augmenting backend API with default type inferer by dynamically
  // creating tensors of input type & executing the op & caching result type.
  return getTensorOrEvalNode().type();
}

bool JitTensorBase::isSparse() {
  return getTensorOrEvalNode().isSparse();
}

Location JitTensorBase::location() {
  // TODO keep track of location to avoid materialization
  return getTensorOrEvalNode().location();
}

void JitTensorBase::scalar(void* out) {
  const auto& tensor = getTensorOrEvalNode();
  switch (type()) {
    case dtype::f16:
      throw std::runtime_error("[JitTensorBase::scalar] f16 unsupported");
    case dtype::f32: *((float*) out) = tensor.scalar<float>(); return;
    case dtype::f64: *((double*) out) = tensor.scalar<double>(); return;
    case dtype::b8: *((char*) out) = tensor.scalar<char>(); return;
    case dtype::s16: *((short*) out) = tensor.scalar<short>(); return;
    case dtype::s32: *((int*) out) = tensor.scalar<int>(); return;
    case dtype::s64: *((long long*) out) = tensor.scalar<long long>(); return;
    case dtype::u8:
      *((unsigned char*) out) = tensor.scalar<unsigned char>();
      return;
    case dtype::u16:
      *((unsigned short*) out) = tensor.scalar<unsigned short>();
      return;
    case dtype::u32:
      *((unsigned int*) out) = tensor.scalar<unsigned int>();
      return;
    case dtype::u64:
      *((unsigned long long*) out) = tensor.scalar<unsigned long long>();
      return;
  }
  throw std::runtime_error("[JitTensorBase::scalar] Unknown data type");
}

void JitTensorBase::device(void** out) {
  getTensorOrEvalNode().device(out);
}

void JitTensorBase::host(void* out) {
  getTensorOrEvalNode().host(out);
}

void JitTensorBase::unlock() {
  getTensorOrEvalNode().unlock();
}

bool JitTensorBase::isLocked() {
  return getTensorOrEvalNode().isLocked();
}

bool JitTensorBase::isContiguous() {
  // TODO does Tensor API semantics allow us to infer contiguity here?
  // e.g., potential sources of discontiguity:
  // 1. indexing (can dynamically check to some extent)
  // 2. ???
  return getTensorOrEvalNode().isContiguous();
}

Shape JitTensorBase::strides() {
  const auto& shape = this->shape();
  std::vector<Dim> strides{1};
  for (int i = 0; i < shape.ndim() - 1; i++) {
    strides.push_back(strides.back() * shape[i]);
  }
  return Shape(strides);
}

const Stream& JitTensorBase::stream() const {
  // TODO
  // 1. how to avoid materialization?
  // 2. consider making `Tensor::stream()` non const
  return const_cast<JitTensorBase*>(this)->getTensorOrEvalNode().stream();
}

// TODO consider making a astype node to eliminate redundant type casting
Tensor JitTensorBase::astype(const dtype type) {
  return fromNode(CustomNode::create(
      "astype",
      { this->node() },
      Shape(this->shape()),
      [=](auto inputs) {
        return inputs.at(0).astype(type);
      }));
}

Tensor JitTensorBase::index(const std::vector<Index>& indices) {
  return fromSharedData(sharedData_, sharedIndexing_->applyIndices(indices));
}

Tensor JitTensorBase::flatten() const {
  return fromNode(CustomNode::create(
      "flatten",
      { this->node() },
      Shape({ node()->shape().elements() }),
      [=](auto inputs) {
        return inputs.at(0).flatten();
      }));
}

Tensor JitTensorBase::flat(const Index& idx) const {
  // TODO shape inference for custom node
  const auto& thisTensorResult =
    const_cast<JitTensorBase*>(this)->getTensorOrEvalNode();
  if (idx.type() == detail::IndexType::Tensor) {
    const auto& tensorIdx = idx.get<Tensor>();
    const auto& tensorIdxResult =
      toJitTensorBase(tensorIdx).getTensorOrEvalNode();
    return fromNode(ValueNode::create(thisTensorResult.flat(tensorIdxResult)));
  }
  return fromNode(ValueNode::create(thisTensorResult.flat(idx)));
}

// TODO consider making a node to allow opt/eval to avoid redundant call
Tensor JitTensorBase::asContiguousTensor() {
  return fromNode(CustomNode::create(
      "asContiguousTensor",
      { this->node() },
      Shape(shape()),
      [=](auto inputs) {
        return inputs.at(0).asContiguousTensor();
      }));
}

void JitTensorBase::setContext(void* /* context */) {
  // no-op
}

void* JitTensorBase::getContext() {
  return nullptr;
}

std::string JitTensorBase::toString() {
  return getTensorOrEvalNode().toString();
}

std::ostream& JitTensorBase::operator<<(std::ostream& ostr) {
  ostr << toString();
  return ostr;
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
    // Again, SSA, but this time we can't fall back to existing ops, i.e., we must
    // increase the expressivity of our IR, thus a new IR node -- `IndexedMerge`.
    // NOTE
    // We let backend can optimize such patterns to avoid redundant memory alloc,
    // e.g., by dispatching to the more efficient `Tensor::[...]assign(...)`
    const auto& indices = sharedIndexing_->getIndex();
    const auto thisDataNode = this->sharedData_->node;
    const auto otherNode = toJitTensorBase(other).node();
    const auto mergeNode = IndexedMergeNode::create(thisDataNode, indices, otherNode);
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
