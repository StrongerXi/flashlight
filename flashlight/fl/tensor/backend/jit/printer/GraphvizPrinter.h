/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_set>

#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/CustomNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/IndexNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/IndexedMergeNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"

namespace fl {

/**
 * A printer that prints a computation graph to a dot graph file.
 */
class GraphvizPrinter {
 public:
  enum class Color {
    Black,
    Green,
    Red,
  };

 private:
  // temporary state during public function call
  std::ostream* os_;
  const std::unordered_map<Node*, float>* nodeToTotTimeMs_{};
  float medianTotTime_ = 0;
  float maxTotTime_ = 0;

  // optionally persistent state
  Color edgeColor_ = Color::Black;
  unsigned subgraphNameCounter_{0};
  unsigned nodeNameCounter_{0};
  // cache names in case nodes are reused over subgraphs
  // TODO concern performance, figure out when one would want to clear cache
  std::unordered_map<Node*, std::string> nodeToName_{};
  std::unordered_set<Node*> edgePrinted_{};

  std::ostream& os();
  std::string generateFreshNodeName();
  std::string generateFreshSubgraphName(const std::string& namePrefix);
  const std::string& getNodeName(Node* node) const;

  void printBinaryNodeLabels(const BinaryNode& node);
  void printCustomNodeLabels(const CustomNode& node);
  void printIndexNodeLabels(const IndexNode& node);
  void printIndexedMergeNodeLabels(const IndexedMergeNode& node);
  std::ostream& printIndices(const std::vector<Index>& indices);
  void printRangeIndex(const range& rangeIdx);
  void printScalarNodeLabels(const ScalarNode& node);
  std::ostream& printScalarValue(const ScalarNode& node);
  void printValueNodeLabels(const ValueNode& node);
  std::ostream& printNodes(Node* node);
  std::ostream& printNodeLabels(Node* node);
  std::ostream& printNodeColor(float tottime);
  std::ostream& printRelativeColor(float tottime);
  std::ostream& printEdges(Node* node);
  std::ostream& printColor(const Color& color);

  void printSubgraph(Node* node, const std::string& namePrefix);

public:
  // no copy/move
  GraphvizPrinter() = default;
  ~GraphvizPrinter();
  GraphvizPrinter(const GraphvizPrinter&) = delete;
  GraphvizPrinter(GraphvizPrinter&&) = delete;
  GraphvizPrinter& operator=(const GraphvizPrinter&) = delete;
  GraphvizPrinter& operator=(GraphvizPrinter&&) = delete;

  // applies to all newly printed edges after this function call
  GraphvizPrinter& setEdgeColor(Color newColor);

  // print the entire computation tree rooted at `node` to given stream as a
  // `subgraph` dot construct, because users might want to look at relationships
  // of nodes across multiple materializations. As a result, client must
  // manually wrap the output with a `digraph/graph` block.
  GraphvizPrinter& printSubgraph(
      Node* node,
      std::ostream& os,
      const std::string& namePrefix);

  // color node with execution statistics
  GraphvizPrinter& printSubgraph(
      Node* node,
      std::ostream& os,
      const std::string& namePrefix,
      const std::unordered_map<Node*, float>& nodeToTotTimeMs_);
};

} // namespace fl
