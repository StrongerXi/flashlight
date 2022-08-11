/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/printer/GraphvizPrinter.h"

#include <cmath>
#include <iomanip>
#include <stdexcept>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/backend/jit/JitTensorBase.h"

namespace fl {

namespace {

const char* binopToStr(const BinaryOp op) {
  switch (op) {
    case BinaryOp::Add:
      return "Add";
    case BinaryOp::Sub:
      return "Sub";
    case BinaryOp::Mul:
      return "Mul";
    case BinaryOp::Div:
      return "Div";
    case BinaryOp::Eq:
      return "Eq";
    case BinaryOp::Neq:
      return "Neq";
    case BinaryOp::Gt:
      return "Gt";
    case BinaryOp::Gte:
      return "Gte";
    case BinaryOp::Lt:
      return "Lt";
    case BinaryOp::Lte:
      return "Lte";
  }
  throw std::runtime_error("Unsupported binary operation type");
}

} // namespace

GraphvizPrinter::~GraphvizPrinter() {
  for (const auto& [node, _] : nodeToName_) {
    node->decRefCount();
  }
  for (const auto& node : edgePrinted_) {
    node->decRefCount();
  }
}

std::ostream& GraphvizPrinter::os() {
  return *os_;
}

std::string GraphvizPrinter::generateFreshNodeName() {
  return "n" + std::to_string(nodeNameCounter_++);
}

std::string GraphvizPrinter::generateFreshSubgraphName(
    const std::string& namePrefix) {
  return namePrefix + "_" + std::to_string(subgraphNameCounter_++);
}

const std::string& GraphvizPrinter::getNodeName(Node* node) const {
  return nodeToName_.at(node);
}

GraphvizPrinter& GraphvizPrinter::setEdgeColor(Color newColor) {
  this->edgeColor_ = newColor;
  return *this;
}

void GraphvizPrinter::printBinaryNodeLabels(const BinaryNode& node) {
  os() << "BinaryNode"
       << "\\n"
       << "op = " << binopToStr(node.op()) << "\\n"
       << "shape = " << node.shape() << "\\n";
}

void GraphvizPrinter::printCustomNodeLabels(const CustomNode& node) {
  os() << node.name() << "\\n"
       << "shape = " << node.shape() << "\\n";
}

void GraphvizPrinter::printIndexNodeLabels(const IndexNode& node) {
  os() << "IndexNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "indices = ";
  printIndices(node.indices()) << "\\n";
}

void GraphvizPrinter::printIndexedMergeNodeLabels(
    const IndexedMergeNode& node) {
  os() << "IndexedMergeNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "indices = ";
  printIndices(node.indices()) << "\\n";
}

void GraphvizPrinter::printRangeIndex(const range& rangeIdx) {
  os() << rangeIdx.start() << ":";
  if (rangeIdx.end().has_value()) {
    os() << rangeIdx.end().value();
  }
  os() << ":" << rangeIdx.stride();
}

std::ostream& GraphvizPrinter::printIndices(const std::vector<Index>& indices) {
  os() << "[";
  for (unsigned i = 0; i < indices.size(); i++) {
    const auto& idx = indices[i];
    switch (idx.type()) {
      case detail::IndexType::Literal:
        os() << idx.get<Dim>();
        break;
      case detail::IndexType::Span:
        os() << ":";
        break;
      case detail::IndexType::Range:
        printRangeIndex(idx.get<range>());
        break;
      case detail::IndexType::Tensor:
        os() << "<Tensor>";
        break;
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
    case dtype::f32:
      os() << node.scalar<float>();
      break;
    case dtype::f64:
      os() << node.scalar<double>();
      break;
    case dtype::b8:
      os() << node.scalar<char>();
      break;
    case dtype::s16:
      os() << node.scalar<short>();
      break;
    case dtype::s32:
      os() << node.scalar<int>();
      break;
    case dtype::s64:
      os() << node.scalar<long long>();
      break;
    case dtype::u8:
      os() << node.scalar<unsigned char>();
      break;
    case dtype::u16:
      os() << node.scalar<unsigned short>();
      break;
    case dtype::u32:
      os() << node.scalar<unsigned int>();
      break;
    case dtype::u64:
      os() << node.scalar<unsigned long long>();
      break;
    default:
      throw std::runtime_error("Unknown data type");
  }
  return os();
}

void GraphvizPrinter::printScalarNodeLabels(const ScalarNode& node) {
  os() << "ScalarNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "dtype = " << node.dataType() << "\\n"
       << "value = ";
  printScalarValue(node) << "\\n";
}

void GraphvizPrinter::printValueNodeLabels(const ValueNode& node) {
  os() << "ValueNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n";
}

std::ostream& GraphvizPrinter::printNodes(Node* node) {
  if (nodeToName_.find(node) == nodeToName_.end()) {
    nodeToName_.emplace(node, generateFreshNodeName());
    node->incRefCount();
    // root at bottom
    for (const auto& input : node->inputs()) {
      printNodes(input);
    }
    os() << "    " << getNodeName(node) << "  [label=\"";
    printNodeLabels(node);
    const auto iter = nodeToTotTimeMs_->find(node);
    if (iter != nodeToTotTimeMs_->end()) {
      const auto tottime = iter->second;
      os() << "tottime = " << tottime << "ms"
           << "\" ";
      printNodeColor(tottime);
    } else { // just end label string
      os() << "\" ";
    }
    os() << "];" << std::endl;
  }
  return os();
}

std::ostream& GraphvizPrinter::printNodeLabels(Node* node) {
  switch (node->type()) {
    case NodeType::Binary:
      printBinaryNodeLabels(node->impl<BinaryNode>());
      break;
    case NodeType::Custom:
      printCustomNodeLabels(node->impl<CustomNode>());
      break;
    case NodeType::Index:
      printIndexNodeLabels(node->impl<IndexNode>());
      break;
    case NodeType::IndexedMerge:
      printIndexedMergeNodeLabels(node->impl<IndexedMergeNode>());
      break;
    case NodeType::Scalar:
      printScalarNodeLabels(node->impl<ScalarNode>());
      break;
    case NodeType::Value:
      printValueNodeLabels(node->impl<ValueNode>());
      break;
    default:
      throw std::runtime_error(
          "[GraphvizPrinter::printNodeLabels] Unknown node type");
  }
  return os();
}

std::ostream& GraphvizPrinter::printNodeColor(float tottime) {
  os() << " fillcolor=\"";
  printRelativeColor(tottime) << "\""
                              << " style=filled ";
  return os();
}

std::ostream& GraphvizPrinter::printRelativeColor(float tottime) {
  // RGB: (why median -- to avoid all red if all nodes have ~= tottime)
  // max    = (255, 0, 0)     -- all read
  // median = (255, 128, 128) -- in between
  // 0      = (255, 255, 255) -- white
  //
  // i.e.,
  // f(0) = 255, f(median) = 128, f(max) = 0
  // We want to differentiate the hotspot more,
  // so f(x) = a(x - b)^2, and we don't try to fit f(0) = 255 -- we just
  // truncate to 255 if result overflows.
  //
  // --> f(max) = a(max - b)^2 = 0
  // Thus b = max
  //
  // --> f(median) = a(median - max)^2 = 128
  // Thus a = 128/(max - median)^2
  //
  // TODO cache (a, b) instead?
  const auto a = 128 / std::pow(maxTotTime_ - medianTotTime_, 2);
  const auto b = maxTotTime_;
  const auto res = std::min(255., a * std::pow(tottime - b, 2));
  const auto normalizedIntensity = static_cast<int>(res);
  std::ios state(NULL);
  state.copyfmt(os());
  os() << "#" << std::hex << 255 << std::setfill('0') << std::setw(2)
       << normalizedIntensity << std::setfill('0') << std::setw(2)
       << normalizedIntensity;
  os().copyfmt(state);
  return os();
}

std::ostream& GraphvizPrinter::printEdges(Node* node) {
  if (edgePrinted_.find(node) == edgePrinted_.end()) {
    edgePrinted_.insert(node);
    node->incRefCount();
    // root at bottom
    for (const auto& input : node->inputs()) {
      printEdges(input);
    }
    for (const auto& input : node->inputs()) {
      os() << "    " << getNodeName(input) << " -> " << getNodeName(node)
           << " ["
           << "color=\"";
      printColor(edgeColor_) << "\""
                             << " ]" << std::endl;
    }
  }
  return os();
}

std::ostream& GraphvizPrinter::printColor(const Color& color) {
  switch (color) {
    case Color::Black:
      return os() << "black";
    case Color::Green:
      return os() << "green";
    case Color::Red:
      return os() << "red";
  }
  throw std::runtime_error("[GraphvizPrinter::printColor] unknown color");
}

void GraphvizPrinter::printSubgraph(Node* node, const std::string& namePrefix) {
  // "cluster_" prefix enables bounding box around subgraph, yep.
  os() << "subgraph cluster_" << generateFreshSubgraphName(namePrefix) << " {"
       << std::endl
       << std::endl;
  printNodes(node) << std::endl;
  printEdges(node) << std::endl;
  os() << "}" << std::endl << std::endl;
}

GraphvizPrinter& GraphvizPrinter::printSubgraph(
    Node* node,
    std::ostream& os,
    const std::string& namePrefix) {
  std::unordered_map<Node*, float> emptyStats;
  return printSubgraph(node, os, namePrefix, emptyStats);
}

GraphvizPrinter& GraphvizPrinter::printSubgraph(
    Node* node,
    std::ostream& os,
    const std::string& namePrefix,
    const std::unordered_map<Node*, float>& nodeToTotTimeMs) {
  // oh boy
  this->os_ = &os;
  this->nodeToTotTimeMs_ = &nodeToTotTimeMs;
  if (!nodeToTotTimeMs.empty()) {
    std::vector<float> vals;
    for (const auto& [_, val] : nodeToTotTimeMs) {
      vals.push_back(val);
    }
    std::sort(vals.begin(), vals.end());
    medianTotTime_ = vals[vals.size() / 2];
    maxTotTime_ = vals.back();
  }
  printSubgraph(node, namePrefix);
  return *this;
}

} // namespace fl
