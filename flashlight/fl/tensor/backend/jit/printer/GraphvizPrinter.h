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
  std::ostream* os_;
  Color edgeColor_ = Color::Black;
  unsigned subgraphNameCounter{0};
  unsigned nodeNameCounter{0};
  // cache names in case nodes are reused over subgraphs
  // TODO concern performance, figure out when one would want to clear cache
  std::unordered_map<std::shared_ptr<Node>, std::string> nodeToName{};
  std::unordered_set<std::shared_ptr<Node>> edgePrinted_{};

  std::ostream& os();
  std::string generateFreshNodeName();
  std::string generateFreshSubgraphName(const std::string& namePrefix);
  const std::string& getNodeName(const std::shared_ptr<Node>& node) const;

  void printBinaryNode(const BinaryNode& node);
  void printCustomNode(const CustomNode& node);
  void printIndexNode(const IndexNode& node);
  void printIndexedMergeNode(const IndexedMergeNode& node);
  std::ostream& printIndices(const std::vector<Index>& indices);
  void printRangeIndex(const range& rangeIdx);
  void printScalarNode(const ScalarNode& node);
  std::ostream& printScalarValue(const ScalarNode& node);
  void printValueNode(const ValueNode& node);
  std::ostream& printNodes(const std::shared_ptr<Node>& node);
  std::ostream& printEdges(const std::shared_ptr<Node>& node);
  std::ostream& printColor(const Color& color);

public:
  // no copy/move
  GraphvizPrinter() = default;
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
      const std::shared_ptr<Node>& node,
      std::ostream& os,
      const std::string& namePrefix);
};

} // namespace fl
