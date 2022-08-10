/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/printer/GraphvizPrinter.h"

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/backend/jit/JitTensorBase.h"

namespace fl {

namespace {

const char* binopToStr(const BinaryOp op) {
  switch(op) {
    case BinaryOp::Add: return "Add";
    case BinaryOp::Sub: return "Sub";
    case BinaryOp::Mul: return "Mul";
    case BinaryOp::Div: return "Div";
    case BinaryOp::Eq: return "Eq";
    case BinaryOp::Neq: return "Neq";
    case BinaryOp::Gt: return "Gt";
    case BinaryOp::Gte: return "Gte";
    case BinaryOp::Lt: return "Lt";
    case BinaryOp::Lte: return "Lte";
  }
  throw std::runtime_error("Unsupported binary operation type");
}

} // namespace

std::ostream& GraphvizPrinter::os() {
  return *os_;
}

std::string GraphvizPrinter::generateFreshNodeName() {
  return "n" + std::to_string(nodeNameCounter++);
}

std::string GraphvizPrinter::generateFreshSubgraphName(
    const std::string& namePrefix) {
  return namePrefix + "_" + std::to_string(subgraphNameCounter++);
}

const std::string& GraphvizPrinter::getNodeName(
    const std::shared_ptr<Node>& node) const {
  return nodeToName.at(node);
}

GraphvizPrinter& GraphvizPrinter::setEdgeColor(Color newColor) {
  this->edgeColor_ = newColor;
  return *this;
}

void GraphvizPrinter::printBinaryNode(const BinaryNode& node) {
  os() << "label=\""
       << "BinaryNode" << "\\n"
       << "op = " << binopToStr(node.op()) << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "\"";
}

void GraphvizPrinter::printCustomNode(const CustomNode& node) {
  os() << "label=\""
      << node.debugName() << "\\n"
      << "shape = " << node.shape() << "\\n"
      << "\"";
}

void GraphvizPrinter::printIndexNode(const IndexNode& node) {
  os() << "label=\""
       << "IndexNode" << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "indices = "; printIndices(node.indices()) << "\\n"
       << "\"";
}

void GraphvizPrinter::printIndexedMergeNode(const IndexedMergeNode& node) {
  os() << "label=\""
       << "IndexedMergeNode" << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "indices = "; printIndices(node.indices()) << "\\n"
       << "\"";
}

void GraphvizPrinter::printRangeIndex(const range& rangeIdx) {
  os() << rangeIdx.start() << ":" << rangeIdx.end() << ":" << rangeIdx.stride();
}

std::ostream& GraphvizPrinter::printIndices(const std::vector<Index>& indices) {
  os() << "[";
  for (unsigned i = 0; i < indices.size(); i++) {
    const auto& idx = indices[i];
    switch (idx.type()) {
      case detail::IndexType::Literal: os() << idx.get<Dim>(); break;
      case detail::IndexType::Span: os() << ":"; break;
      case detail::IndexType::Range: printRangeIndex(idx.get<range>()); break;
      case detail::IndexType::Tensor: os() << "<Tensor>"; break;
    }
    if (i != indices.size() - 1) {
      os() << ", ";
    }
  }
  return os();
}

std::ostream& GraphvizPrinter::printScalarValue(const ScalarNode& node) {
  switch (node.dataType()) {
    case dtype::f16:
      throw std::runtime_error(
          "[GraphvizPrinter::printScalarNodeValue] f16 is unsupported");
    case dtype::f32: os() << node.scalar<float>(); break;
    case dtype::f64: os() << node.scalar<double>(); break;
    case dtype::b8: os() << node.scalar<char>(); break;
    case dtype::s16: os() << node.scalar<short>(); break;
    case dtype::s32: os() << node.scalar<int>(); break;
    case dtype::s64: os() << node.scalar<long long>(); break;
    case dtype::u8: os() << node.scalar<unsigned char>(); break;
    case dtype::u16: os() << node.scalar<unsigned short>(); break;
    case dtype::u32: os() << node.scalar<unsigned int>(); break;
    case dtype::u64: os() << node.scalar<unsigned long long>(); break;
    default: throw std::runtime_error("Unknown data type");
  }
  return os();
}

void GraphvizPrinter::printScalarNode(const ScalarNode& node) {
  os() << "label=\""
       << "ScalarNode" << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "dtype = " << node.dataType() << "\\n"
       << "value = "; printScalarValue(node) << "\\n"
       << "\"";
}

void GraphvizPrinter::printValueNode(const ValueNode& node) {
  os() << "label=\""
       << "ValueNode" << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "\"";
}

std::ostream& GraphvizPrinter::printNodes(const std::shared_ptr<Node>& node) {
  if (nodeToName.find(node) == nodeToName.end()) {
    nodeToName.emplace(node, generateFreshNodeName());
    // root at bottom
    for (const auto& input : node->inputs()) {
      printNodes(input);
    }
    os() << "    " << getNodeName(node) << "  [";
    switch (node->type()) {
      case NodeType::Binary:
        printBinaryNode(node->impl<BinaryNode>()); break;
      case NodeType::Custom:
        printCustomNode(node->impl<CustomNode>()); break;
      case NodeType::Index:
        printIndexNode(node->impl<IndexNode>()); break;
      case NodeType::IndexedMerge:
        printIndexedMergeNode(node->impl<IndexedMergeNode>()); break;
      case NodeType::Scalar:
        printScalarNode(node->impl<ScalarNode>()); break;
      case NodeType::Value:
        printValueNode(node->impl<ValueNode>()); break;
    }
    os() << "];" << std::endl;
  }
  return os();
}

std::ostream& GraphvizPrinter::printEdges(const std::shared_ptr<Node>& node) {
  if (edgePrinted_.find(node) == edgePrinted_.end()) {
    edgePrinted_.insert(node);
    // root at bottom
    for (const auto& input : node->inputs()) {
      printEdges(input);
    }
    for (const auto& input : node->inputs()) {
      os() << "    "
          << getNodeName(node) << " -> " << getNodeName(input)
          << " ["
          << "color=\""; printColor(edgeColor_) << "\""
          << " ]"
          << std::endl;
    }
  }
  return os();
}

std::ostream& GraphvizPrinter::printColor(const Color& color) {
  switch (color) {
    case Color::Black: return os() << "black";
    case Color::Green: return os() << "green";
    case Color::Red: return os() << "red";
  }
  throw std::runtime_error("[GraphvizPrinter::printColor] unknown color");
}

GraphvizPrinter& GraphvizPrinter::printSubgraph(
    const std::shared_ptr<Node>& node,
    std::ostream& os,
    const std::string& namePrefix) {
  this->os_ = &os; // oh boy
  os << "subgraph " << generateFreshSubgraphName(namePrefix) << " {" << std::endl
     << std::endl;
  printNodes(node) << std::endl;
  printEdges(node) << std::endl;
  os << "}" << std::endl
     << std::endl;
  return *this;
}

} // namespace fl
