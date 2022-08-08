/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/JitBackend.h"

#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <iostream>

#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/jit/JitTensorBase.h"
#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"

namespace fl {

namespace {

std::vector<std::shared_ptr<Node>> tensorsToNodes(
    const std::vector<Tensor>& tensors) {
  std::vector<std::shared_ptr<Node>> nodes;
  // JitTensor copy is ~free, but whatever
  for (const auto& tensor : tensors) {
    nodes.push_back(toJitTensorBase(tensor).node());
  }
  return nodes;
}

template <typename... T>
std::vector<std::shared_ptr<Node>> tensorsToNodes(const T&... tensors) {
  std::vector<std::shared_ptr<Node>> nodes;
  // JitTensor copy is ~free, but whatever
  for (const auto& tensor : { &tensors... } ) {
    nodes.push_back(toJitTensorBase(*tensor).node());
  }
  return nodes;
}

template <>
std::vector<std::shared_ptr<Node>> tensorsToNodes() {
  return {};
}

const Tensor& materialize(Tensor tensor) {
  auto& jitTensor = toJitTensorBase(tensor);
  jitTensor.eval();
  return jitTensor.node()->getResult().value();
}

// TODO refactor with or reuse logic in OneDnnBackend
Shape inferReductionOutputShape(
    const Shape& inputShape,
    const std::vector<int>& axes,
    bool keepDims) {
  for (const auto axis : axes) {
    if (axis < 0 || axis >= inputShape.ndim()) {
      std::ostringstream oss;
      oss << "[inferReductionOutputShape] Invalid axis for reduction: " << axis
          << " for tensor of shape: " << inputShape;
      throw std::invalid_argument(oss.str());
    }
  }
  std::unordered_set<int> axesToReduce;
  if (axes.empty()) {
    for (int aixs = 0; aixs < inputShape.ndim(); aixs++) {
      axesToReduce.insert(aixs);
    }
  } else {
    axesToReduce.insert(axes.begin(), axes.end());
  }
  std::vector<Dim> outputDims;
  for (int axis = 0; axis < inputShape.ndim(); axis++) {
    if (axesToReduce.find(axis) != axesToReduce.end()) {
      if (keepDims) {
        outputDims.push_back(1);
      }
    } else {
      outputDims.push_back(inputShape.dim(axis));
    }
  }
  return Shape(outputDims);
}

} // namespace

JitBackend::JitBackend(
    TensorBackend& wrappedBackend,
    std::function<Tensor(std::shared_ptr<Node>)> jitTensorCreator)
    : wrappedBackend_(wrappedBackend), jitTensorCreator_(jitTensorCreator) {}

TensorBackendType JitBackend::backendType() const {
  // Implementers of a backend should create their own option in the
  // TensorBackendType enum and return it here.
  return TensorBackendType::Jit;
}

/* -------------------------- Compute Functions -------------------------- */

void JitBackend::eval(const Tensor& tensor) {
  toJitTensorBase(tensor).eval();
}

bool JitBackend::supportsDataType(const fl::dtype& dtype) const {
  return wrappedBackend_.supportsDataType(dtype);
}

void JitBackend::getMemMgrInfo(
    const char* msg,
    const int deviceId,
    std::ostream* ostream) {
  return wrappedBackend_.getMemMgrInfo(msg, deviceId, ostream);
}

void JitBackend::setMemMgrLogStream(std::ostream* stream) {
  return wrappedBackend_.setMemMgrLogStream(stream);
}

void JitBackend::setMemMgrLoggingEnabled(const bool enabled) {
  return wrappedBackend_.setMemMgrLoggingEnabled(enabled);
}

void JitBackend::setMemMgrFlushInterval(const size_t interval) {
  return wrappedBackend_.setMemMgrFlushInterval(interval);
}

/* -------------------------- Rand Functions -------------------------- */

void JitBackend::setSeed(const int seed) {
  wrappedBackend_.setSeed(seed);
}

Tensor JitBackend::randn(const Shape& shape, dtype type) {
  return jitTensorCreator_(CustomNode::create(
      "randn",
      tensorsToNodes(),
      Shape(shape),
      [=](auto /* inputs */) {
        return wrappedBackend_.randn(shape, type);
      }));
}

Tensor JitBackend::rand(const Shape& shape, dtype type) {
  return jitTensorCreator_(CustomNode::create(
      "rand",
      tensorsToNodes(),
      Shape(shape),
      [=](auto /* inputs */) {
        return wrappedBackend_.rand(shape, type);
      }));
}

/* --------------------------- Tensor Operators --------------------------- */

/******************** Tensor Creation Functions ********************/
#define FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(TYPE)                      \
  Tensor JitBackend::fromScalar(TYPE /* value */, const dtype /* type */) {   \
    throw std::invalid_argument(                                              \
        "JitBackend::fromScalar - not implemented for type " +                \
        std::string(#TYPE));                                                  \
  }                                                                           \
  Tensor JitBackend::full(const Shape& shape, TYPE value, const dtype type) { \
    switch (type) {                                                           \
      case dtype::f16:                                                        \
        return fullWithType(shape, value, dtype::f32).astype(dtype::f16);     \
      case dtype::f32: return fullWithType(shape, value, type);               \
      case dtype::f64: return fullWithType(shape, value, type);               \
      case dtype::b8: return fullWithType(shape, value, type);                \
      case dtype::s16: return fullWithType(shape, value, type);               \
      case dtype::s32: return fullWithType(shape, value, type);               \
      case dtype::s64: return fullWithType(shape, value, type);               \
      case dtype::u8: return fullWithType(shape, value, type);                \
      case dtype::u16: return fullWithType(shape, value, type);               \
      case dtype::u32: return fullWithType(shape, value, type);               \
      case dtype::u64: return fullWithType(shape, value, type);               \
    }                                                                         \
    throw std::runtime_error("Unknown dtype");                                \
  }
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const double&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const float&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const int&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const unsigned&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const char&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const unsigned char&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const long&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const unsigned long&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const long long&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const unsigned long long&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const bool&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const short&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const unsigned short&);

template<typename T>
Tensor JitBackend::fullWithType(const Shape& shape, T value, dtype type) {
  return jitTensorCreator_(ScalarNode::create(Shape(shape), type, value));
}

Tensor JitBackend::identity(const Dim dim, const dtype type) {
  return
    jitTensorCreator_(ValueNode::create(wrappedBackend_.identity(dim, type)));
}

Tensor JitBackend::arange(
    const Shape& shape,
    const Dim seqDim,
    const dtype type) {
  return jitTensorCreator_(CustomNode::create(
      "arange",
      tensorsToNodes(),
      Shape(shape),
      [=](auto /* inputs */) {
        return wrappedBackend_.arange(shape, seqDim, type);
      }));
}

Tensor JitBackend::iota(
    const Shape& dims,
    const Shape& tileDims,
    const dtype type) {
  return jitTensorCreator_(
      ValueNode::create(wrappedBackend_.iota(dims, tileDims, type)));
}

/************************ Shaping and Indexing *************************/
Tensor JitBackend::reshape(
    const Tensor& tensor,
    const Shape& shape) {
  return jitTensorCreator_(CustomNode::create(
      "reshape",
      tensorsToNodes(tensor),
      Shape(shape),
      [=](auto inputs) {
        return wrappedBackend_.reshape(inputs.at(0), shape);
      }));
}

Tensor JitBackend::transpose(
    const Tensor& tensor,
    const Shape& axes = {}) {
  // TODO refactor with logic in OneDnnBackend
  Shape newShape = tensor.shape();
  std::vector<Dim> oldToNewAxes = axes.get();
  if (axes.ndim() == 0) { // default, reverse all axes
    oldToNewAxes.resize(tensor.ndim());
    std::reverse(newShape.get().begin(), newShape.get().end());
    std::iota(oldToNewAxes.begin(), oldToNewAxes.end(), 0);
    std::reverse(oldToNewAxes.begin(), oldToNewAxes.end());
  } else if (axes.ndim() == tensor.ndim()) {
    for (int axis = 0; axis < axes.ndim(); axis++) {
      newShape[axis] = tensor.dim(oldToNewAxes[axis]);
    }
  } else {
    std::invalid_argument(
        "[JitBackend::transpose] Invalid axes: " + axes.toString() +
        " for shape: " + tensor.shape().toString());
  }
  return jitTensorCreator_(CustomNode::create(
      "transpose",
      tensorsToNodes(tensor),
      Shape(newShape),
      [=](auto inputs) {
        return wrappedBackend_.reshape(inputs.at(0), axes);
      }));
}

Tensor JitBackend::tile(const Tensor& tensor, const Shape& tileDims) {
  // TODO refactor with logic in OneDnnBackend
  std::vector<Dim> paddedTensorDims = tensor.shape().get();
  std::vector<Dim> paddedTileDims = tileDims.get();
  const auto tensorShapeNDims = tensor.ndim();
  const auto tileDimsNDims = tileDims.ndim();
  if (tensorShapeNDims > tileDimsNDims) {
    const auto diff = tensorShapeNDims - tileDimsNDims;
    paddedTileDims.insert(paddedTileDims.end(), diff, 1);
  } else {
    const auto diff = tileDimsNDims - tensorShapeNDims;
    paddedTensorDims.insert(paddedTensorDims.end(), diff, 1);
  }
  std::vector<Dim> outputDims;
  for (int i = 0; i < paddedTensorDims.size(); i++) {
    outputDims.push_back(paddedTensorDims[i] * paddedTileDims[i]);
  }
  return jitTensorCreator_(CustomNode::create(
      "tile",
      tensorsToNodes(tensor),
      Shape(outputDims),
      [=](auto /* inputs */) {
        return wrappedBackend_.tile(tensor, tileDims);
      }));
}

Tensor JitBackend::concatenate(
    const std::vector<Tensor>& tensors,
    const unsigned axisToConcat) {
  // TODO need a nice way to construct empty tensor for wrapped backend
  if (tensors.empty()) {
    throw std::runtime_error(
        "[JitBackend::concatenate] Nothing to concatenate");
  }
  const auto& shape = tensors.front().shape();
  const unsigned ndim = shape.ndim();
  for (unsigned i = 1; i < tensors.size(); i++) {
    for (unsigned axis = 0; axis < ndim; axis++) {
      if (axis != axisToConcat && shape[axis] != tensors[i].dim(axis)) {
        throw std::runtime_error(
            "[JitBackend::concatenate] Broadcasting is unsupported");
      }
    }
  }
  std::vector<Dim> outputDims;
  for (unsigned axis = 0; axis < ndim; axis++) {
    if (axis == axisToConcat) {
      outputDims.push_back(tensors.size() * shape[axis]);
    } else {
      outputDims.push_back(shape[axis]);
    }
  }
  return jitTensorCreator_(CustomNode::create(
      "concatenate",
      tensorsToNodes(tensors),
      Shape(outputDims),
      [=](auto inputs) {
        return wrappedBackend_.concatenate(inputs, axisToConcat);
      }));
}

Tensor JitBackend::nonzero(const Tensor& tensor) {
  // TODO the benefit of retaining a graph in a `nonzero` node doesn't seem to
  // justify the potential engineering effort. Is this ever used as an
  // intermediate computation?
  const auto& tensorResult = materialize(tensor);
  return
    jitTensorCreator_(ValueNode::create(wrappedBackend_.nonzero(tensorResult)));
}

Tensor JitBackend::pad(
    const Tensor& input,
    const std::vector<std::pair<int, int>>& padWidths,
    const PadType type) {
  std::vector<Dim> outputDims = input.shape().get();
  if (padWidths.size() > static_cast<size_t>(input.ndim())) {
    throw std::runtime_error("[JitBackend::pad] too many paddings");
  }
  for (unsigned axis = 0; axis < padWidths.size(); axis++) {
    const auto& [beforeDim, afterDim] = padWidths[axis];
    outputDims[axis] += beforeDim + afterDim;
  }
  return jitTensorCreator_(CustomNode::create(
      "pad",
      tensorsToNodes(input),
      Shape(outputDims),
      [=](auto inputs) {
        return wrappedBackend_.pad(inputs.at(0), padWidths, type);
      }));
}

/************************** Unary Operators ***************************/

#define FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(OP) { \
  return jitTensorCreator_(CustomNode::create(   \
      #OP,                                       \
      tensorsToNodes(tensor),                    \
      Shape(tensor.shape()),                     \
      [=](auto inputs) {                         \
        return wrappedBackend_.OP(inputs.at(0)); \
      }));                                       \
}

Tensor JitBackend::exp(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(exp);
}

Tensor JitBackend::log(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(log);
}

Tensor JitBackend::negative(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(negative);
}

Tensor JitBackend::logicalNot(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(logicalNot);
}

Tensor JitBackend::log1p(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(log1p);
}

Tensor JitBackend::sin(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(sin);
}

Tensor JitBackend::cos(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(cos);
}

Tensor JitBackend::sqrt(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(sqrt);
}

Tensor JitBackend::tanh(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(tanh);
}

Tensor JitBackend::floor(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(floor);
}

Tensor JitBackend::ceil(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(ceil);
}

Tensor JitBackend::rint(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(rint);
}

Tensor JitBackend::absolute(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(absolute);
}

Tensor JitBackend::sigmoid(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(sigmoid);
}

Tensor JitBackend::erf(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(erf);
}

Tensor JitBackend::flip(const Tensor& tensor, const unsigned dim) {
  return jitTensorCreator_(CustomNode::create(
      "flip",
      tensorsToNodes(tensor),
      Shape(tensor.shape()),
      [=](auto inputs) {
        return wrappedBackend_.flip(inputs.at(0), dim);
      }));
}

Tensor JitBackend::clip(
    const Tensor& tensor,
    const Tensor& low,
    const Tensor& high) {
  return jitTensorCreator_(CustomNode::create(
      "clip",
      tensorsToNodes(tensor, low, high),
      Shape(tensor.shape()),
      [=](auto inputs) {
        return wrappedBackend_.clip(inputs.at(0), inputs.at(1), inputs.at(2));
      }));
}

Tensor JitBackend::roll(
    const Tensor& tensor,
    const int shift,
    const unsigned axis) {
  return jitTensorCreator_(CustomNode::create(
      "roll",
      tensorsToNodes(tensor),
      Shape(tensor.shape()),
      [=](auto inputs) {
        return wrappedBackend_.roll(inputs.at(0), shift, axis);
      }));
}

Tensor JitBackend::isnan(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(isnan);
}

Tensor JitBackend::isinf(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(isinf);
}

Tensor JitBackend::sign(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(sign);
}

Tensor JitBackend::tril(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(tril);
}

Tensor JitBackend::triu(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(triu);
}
#undef FL_JIT_BACKEND_UNARY_FALLBACK_IMPL

Tensor JitBackend::where(
    const Tensor& condition,
    const Tensor& x,
    const Tensor& y) {
  return jitTensorCreator_(CustomNode::create(
      "where",
      tensorsToNodes(condition, x, y),
      Shape(condition.shape()),
      [=](auto inputs) {
        return wrappedBackend_.where(inputs.at(0), inputs.at(1), inputs.at(2));
      }));
}

void JitBackend::topk(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned k,
    const Dim axis,
    const SortMode sortMode) {
  const auto& inputResult = materialize(input);
  auto valuesResult = this->full({1}, 0, dtype::s32);
  auto indicesResult = this->full({1}, 0, dtype::s32);
  wrappedBackend_.topk(
      valuesResult, indicesResult, inputResult, k, axis, sortMode);
  values = jitTensorCreator_(ValueNode::create(std::move(valuesResult)));
  indices = jitTensorCreator_(ValueNode::create(std::move(indicesResult)));
}

Tensor JitBackend::sort(
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode) {
  return jitTensorCreator_(CustomNode::create(
      "sort",
      tensorsToNodes(input),
      Shape(input.shape()),
      [=](auto inputs) {
        return wrappedBackend_.sort(inputs.at(0), axis, sortMode);
      }));
}

void JitBackend::sort(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode) {
  const auto& inputResult = materialize(input);
  auto valuesResult = this->full({1}, 0, dtype::s32);
  auto indicesResult = this->full({1}, 0, dtype::s32);
  wrappedBackend_.sort(
      valuesResult, indicesResult, inputResult, axis, sortMode);
  values = jitTensorCreator_(ValueNode::create(std::move(valuesResult)));
  indices = jitTensorCreator_(ValueNode::create(std::move(indicesResult)));
}

Tensor JitBackend::argsort(
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode) {
  return jitTensorCreator_(CustomNode::create(
      "argsort",
      tensorsToNodes(input),
      Shape(input.shape()),
      [=](auto inputs) {
        return wrappedBackend_.argsort(inputs.at(0), axis, sortMode);
      }));
}

/************************** Binary Operators ***************************/
#define FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, TYPE)                      \
  Tensor JitBackend::FUNC(const Tensor& /* a */, TYPE /* rhs */) {          \
    throw std::runtime_error(                                               \
        "JitBackend::" + std::string(#FUNC) + " unimplemented for type " +  \
        std::string(#TYPE));                                                \
  }                                                                         \
  Tensor JitBackend::FUNC(TYPE /* lhs */, const Tensor& /* a */) {          \
    throw std::runtime_error(                                               \
        "JitBackend::" + std::string(#FUNC) + " unimplemented for type " +  \
        std::string(#TYPE));                                                \
  }

#define FL_JIT_BINARY_OP_LITERALS_DEF_STUB(FUNC, OP)                   \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const bool&);               \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const int&);                \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const unsigned&);           \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const char&);               \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const unsigned char&);      \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const long&);               \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const unsigned long&);      \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const long long&);          \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const unsigned long long&); \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const double&);             \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const float&);              \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const short&);              \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const unsigned short&);

// Operations on fl::Tensor call the respective operator overloads that are
// already defined on af::arrays
#define FL_JIT_BINARY_OP_DEF_STUB(OP, FUNC)                                    \
  Tensor JitBackend::FUNC(const Tensor& /* lhs */, const Tensor& /* rhs */) {  \
    throw std::runtime_error(                                                  \
        "JitBackend::" + std::string(#FUNC) +                                  \
        " unimplemented for two-Tensor inputs.");                              \
  }                                                                            \
  FL_JIT_BINARY_OP_LITERALS_DEF_STUB(FUNC, OP);

FL_JIT_BINARY_OP_LITERALS_DEF_STUB(logicalOr, ||);
FL_JIT_BINARY_OP_LITERALS_DEF_STUB(logicalAnd, &&);
FL_JIT_BINARY_OP_DEF_STUB(%, mod);
FL_JIT_BINARY_OP_DEF_STUB(&, bitwiseAnd);
FL_JIT_BINARY_OP_DEF_STUB(|, bitwiseOr);
FL_JIT_BINARY_OP_DEF_STUB(^, bitwiseXor);
FL_JIT_BINARY_OP_DEF_STUB(<<, lShift);
FL_JIT_BINARY_OP_DEF_STUB(>>, rShift);
#undef FL_JIT_BINARY_OP_DEF
#undef FL_JIT_BINARY_OP_TYPE_DEF
#undef FL_JIT_BINARY_OP_LITERALS_DEF

#define FL_JIT_BINARY_OP_TYPE_DEF(FUNC, TYPE)                   \
  Tensor JitBackend::FUNC(const Tensor& a, TYPE rhs) {          \
    const auto dtype = dtype_traits<std::decay_t<TYPE>>::ctype; \
    return FUNC(a, this->full(a.shape(), rhs, dtype));          \
  }                                                             \
  Tensor JitBackend::FUNC(TYPE lhs, const Tensor& a) {          \
    const auto dtype = dtype_traits<std::decay_t<TYPE>>::ctype; \
    return FUNC(this->full(a.shape(), lhs, dtype), a);          \
  }                                                             \

#define FL_JIT_BINARY_OP_LITERALS_DEF(FUNC)                   \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const bool&);               \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const int&);                \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const unsigned&);           \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const char&);               \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const unsigned char&);      \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const long&);               \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const unsigned long&);      \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const long long&);          \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const unsigned long long&); \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const double&);             \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const float&);              \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const short&);              \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const unsigned short&);

#define FL_JIT_BINARY_OP_TENSOR_DEF(FUNC, BINOP)                           \
  Tensor JitBackend::FUNC(const Tensor& lhs, const Tensor& rhs) {          \
    const auto lhsNode = toJitTensorBase(lhs).node();                      \
    const auto rhsNode = toJitTensorBase(rhs).node();                      \
    return jitTensorCreator_(BinaryNode::create(lhsNode, rhsNode, BINOP)); \
  }                                                                        \
  FL_JIT_BINARY_OP_LITERALS_DEF(FUNC);

FL_JIT_BINARY_OP_TENSOR_DEF(add, BinaryOp::Add);
FL_JIT_BINARY_OP_TENSOR_DEF(sub, BinaryOp::Sub);
FL_JIT_BINARY_OP_TENSOR_DEF(mul, BinaryOp::Mul);
FL_JIT_BINARY_OP_TENSOR_DEF(div, BinaryOp::Div);
FL_JIT_BINARY_OP_TENSOR_DEF(eq, BinaryOp::Eq);
FL_JIT_BINARY_OP_TENSOR_DEF(neq, BinaryOp::Neq);
FL_JIT_BINARY_OP_TENSOR_DEF(lessThan, BinaryOp::Lt);
FL_JIT_BINARY_OP_TENSOR_DEF(lessThanEqual, BinaryOp::Lte);
FL_JIT_BINARY_OP_TENSOR_DEF(greaterThan, BinaryOp::Gt);
FL_JIT_BINARY_OP_TENSOR_DEF(greaterThanEqual, BinaryOp::Gte);
#undef FL_JIT_BINARY_OP_TYPE_DEF
#undef FL_JIT_BINARY_OP_LITERALS_DEF
#undef FL_JIT_BINARY_OP_TENSOR_DEF

#define FL_JIT_BACKEND_BINARY_FALLBACK_IMPL(OP) {                 \
  if (lhs.shape() != rhs.shape()) {                               \
    throw std::runtime_error(                                     \
        "[JitBackend] Currently no support for binop broadcast"); \
  }                                                               \
  return jitTensorCreator_(CustomNode::create(                    \
      #OP,                                                        \
      tensorsToNodes(lhs, rhs),                                   \
      Shape(lhs.shape()),                                         \
      [=](auto inputs) {                                          \
        return wrappedBackend_.OP(inputs.at(0), inputs.at(0));    \
      }));                                                        \
}

Tensor JitBackend::logicalAnd(const Tensor& lhs, const Tensor& rhs) {
  FL_JIT_BACKEND_BINARY_FALLBACK_IMPL(logicalAnd);
}

Tensor JitBackend::logicalOr(const Tensor& lhs, const Tensor& rhs) {
  FL_JIT_BACKEND_BINARY_FALLBACK_IMPL(logicalOr);
}

Tensor JitBackend::minimum(const Tensor& lhs, const Tensor& rhs) {
  FL_JIT_BACKEND_BINARY_FALLBACK_IMPL(minimum);
}

Tensor JitBackend::maximum(const Tensor& lhs, const Tensor& rhs) {
  FL_JIT_BACKEND_BINARY_FALLBACK_IMPL(maximum);
}

Tensor JitBackend::power(const Tensor& lhs, const Tensor& rhs) {
  FL_JIT_BACKEND_BINARY_FALLBACK_IMPL(power);
}
#undef FL_JIT_BACKEND_BINARY_FALLBACK_IMPL

/************************** BLAS ***************************/

Tensor JitBackend::matmul(
    const Tensor& lhs,
    const Tensor& rhs,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp) {
  // TODO refactor with logic in OneDnnBackend
  std::vector<Dim> lhsDims = lhs.shape().get();
  std::vector<Dim> rhsDims = rhs.shape().get();
  const bool isLhsScalarOrVector = lhsDims.size() <= 1;
  const bool isRhsScalarOrVector = rhsDims.size() <= 1;
  if (isLhsScalarOrVector) { // pad to (1 x 1/K)
    lhsDims.insert(lhsDims.end(), 2 - lhsDims.size(), 1);
    std::reverse(lhsDims.begin(), lhsDims.end());
  } else if (lhsProp == MatrixProperty::Transpose) {
    std::swap(lhsDims[0], lhsDims[1]);
  }
  if (isRhsScalarOrVector) { // pad to (1/K x 1)
    rhsDims.insert(rhsDims.end(), 2 - rhsDims.size(), 1);
  } else if (rhsProp == MatrixProperty::Transpose) {
    std::swap(rhsDims[0], rhsDims[1]);
  }
  // shape check (TODO support broadcasting)
  if (!(lhsDims.at(1) == rhsDims.at(0) &&
        std::equal(
          lhsDims.begin() + 2,
          lhsDims.end(),
          rhsDims.begin() + 2,
          rhsDims.end()))) {
    std::ostringstream oss;
    oss << "Cannot perform matmul for tensors of shapes: " << lhs.shape()
      << " and " << rhs.shape();
    throw std::invalid_argument(oss.str());
  }
  std::vector<Dim> outputDims = lhsDims;
  outputDims[1] = rhsDims[1];
  Shape outputShape(outputDims);
  if (isLhsScalarOrVector || isRhsScalarOrVector) {
    outputShape = {outputShape.elements()};
  }
  return jitTensorCreator_(CustomNode::create(
      "matmul",
      tensorsToNodes(lhs, rhs),
      Shape(outputShape),
      [=](auto inputs) {
        return wrappedBackend_.matmul(inputs.at(0), inputs.at(1), lhsProp, rhsProp);
      }));
}

/************************** Reductions ***************************/

#define FL_JIT_BACKEND_REDUCTION_FALLBACK_IMPL(OP) {             \
  return jitTensorCreator_(CustomNode::create(                   \
      #OP,                                                       \
      tensorsToNodes(input),                                     \
      inferReductionOutputShape(input.shape(), axes, keepDims),  \
      [=](auto inputs) {                                         \
        return wrappedBackend_.OP(inputs.at(0), axes, keepDims); \
      }));                                                       \
}

Tensor JitBackend::amin(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  FL_JIT_BACKEND_REDUCTION_FALLBACK_IMPL(amin);
}

Tensor JitBackend::amax(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  FL_JIT_BACKEND_REDUCTION_FALLBACK_IMPL(amax);
}

void JitBackend::min(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims) {
  // TODO consider changing the Tensor API to return a tuple of tensor instead
  // of using an "output parameter".
  const auto& inputResult = materialize(input);
  auto valuesResult = this->full({1}, 0, dtype::s32);
  auto indicesResult = this->full({1}, 0, dtype::s32);
  wrappedBackend_.min(
      valuesResult, indicesResult, inputResult, axis, keepDims);
  values = jitTensorCreator_(ValueNode::create(std::move(valuesResult)));
  indices = jitTensorCreator_(ValueNode::create(std::move(indicesResult)));
}

void JitBackend::max(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims) {
  const auto& inputResult = materialize(input);
  auto valuesResult = this->full({1}, 0, dtype::s32);
  auto indicesResult = this->full({1}, 0, dtype::s32);
  wrappedBackend_.max(
      valuesResult, indicesResult, inputResult, axis, keepDims);
  values = jitTensorCreator_(ValueNode::create(std::move(valuesResult)));
  indices = jitTensorCreator_(ValueNode::create(std::move(indicesResult)));
}

Tensor JitBackend::sum(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  FL_JIT_BACKEND_REDUCTION_FALLBACK_IMPL(sum);
}

Tensor JitBackend::cumsum(
    const Tensor& input,
    const unsigned axis) {
  return jitTensorCreator_(CustomNode::create(
      "cumsum",
      tensorsToNodes(input),
      Shape(input.shape()),
      [=](auto inputs) {
        return wrappedBackend_.cumsum(inputs.at(0), axis);
      }));

}

Tensor JitBackend::argmax(
    const Tensor& input,
    const unsigned axis,
    const bool keepDims) {
  return jitTensorCreator_(CustomNode::create(
      "argmax",
      tensorsToNodes(input),
      inferReductionOutputShape(input.shape(), { static_cast<int>(axis) }, keepDims),
      [=](auto inputs) {
        return wrappedBackend_.argmax(inputs.at(0), axis, keepDims);
      }));
}

Tensor JitBackend::argmin(
    const Tensor& input,
    const unsigned axis,
    const bool keepDims) {
  return jitTensorCreator_(CustomNode::create(
      "argmax",
      tensorsToNodes(input),
      inferReductionOutputShape(input.shape(), { static_cast<int>(axis) }, keepDims),
      [=](auto inputs) {
        return wrappedBackend_.argmax(inputs.at(0), axis, keepDims);
      }));
}

Tensor JitBackend::mean(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  FL_JIT_BACKEND_REDUCTION_FALLBACK_IMPL(mean);
}

Tensor JitBackend::median(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  FL_JIT_BACKEND_REDUCTION_FALLBACK_IMPL(median);
}

Tensor JitBackend::var(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool bias,
    const bool keepDims) {
  return jitTensorCreator_(CustomNode::create(
      "var",
      tensorsToNodes(input),
      inferReductionOutputShape(input.shape(), axes, keepDims),
      [=](auto inputs) {
        return wrappedBackend_.var(inputs.at(0), axes, bias, keepDims);
      }));
}

Tensor JitBackend::std(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  FL_JIT_BACKEND_REDUCTION_FALLBACK_IMPL(std);
}

Tensor JitBackend::norm(
    const Tensor& input,
    const std::vector<int>& axes,
    double p /* = 2 */,
    const bool keepDims) {
  return jitTensorCreator_(CustomNode::create(
      "norm",
      tensorsToNodes(input),
      inferReductionOutputShape(input.shape(), axes, keepDims),
      [=](auto inputs) {
        return wrappedBackend_.norm(inputs.at(0), axes, p, keepDims);
      }));
}

Tensor JitBackend::countNonzero(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  FL_JIT_BACKEND_REDUCTION_FALLBACK_IMPL(countNonzero);
}

Tensor JitBackend::any(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  FL_JIT_BACKEND_REDUCTION_FALLBACK_IMPL(any);
}

Tensor JitBackend::all(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  FL_JIT_BACKEND_REDUCTION_FALLBACK_IMPL(all);
}

void JitBackend::print(const Tensor& tensor) {
  std::cout << "JitTensor" << std::endl
            << toJitTensorBase(tensor).toString() << std::endl;
}

} // namespace fl
