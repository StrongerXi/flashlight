/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#include "flashlight/fl/tensor/backend/jit/JitTensor.h"
#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"
#include "flashlight/fl/tensor/backend/onednn/OneDnnTensor.h"

#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>

using fl::ArrayFireTensor;
using fl::JitTensor;
using fl::OneDnnTensor;
using fl::ScalarNode;
using fl::Shape;
using fl::dtype;

void time(std::function<void()> func) {
  using milli = std::chrono::milliseconds;
  auto start = std::chrono::high_resolution_clock::now();

  func();

  auto finish = std::chrono::high_resolution_clock::now();
  std::cout << "Took "
            << std::chrono::duration_cast<milli>(finish - start).count()
            << " milliseconds\n";
}

//TEST(FooTest, constantFolding) {
//  auto scalarSum = []() {
//    fl::Shape shape({100, 100});
//    auto iter = 100;
//    auto sum = fl::full(shape, 1, dtype::s32);
//    for (int i = 0; i < iter; i++) {
//      sum = sum + fl::full(shape, 1, dtype::s32);
//    }
//    fl::eval(sum);
//  };
//
//  fl::setDefaultTensorType<OneDnnTensor>();
//  time(scalarSum);
//
//  fl::setDefaultTensorType<JitTensor<OneDnnTensor>>();
//  time(scalarSum);
//}

TEST(FooTest, binopFusionTree) {
  std::vector<fl::Tensor> inputs;

  // easy to enable JIT for any backend
  //fl::setDefaultTensorType<OneDnnTensor>();            // ~300ms
  //fl::setDefaultTensorType<JitTensor<OneDnnTensor>>(); // ~15ms
  auto iters = 100;
  // ... omitted data init ...
  // benchmark binops only, not input data allocation
  auto nonscalarSum = [&]() {
    auto sum = inputs[0];
    for (int i = 1; i + 4 <= iters; i += 4) {
      sum = sum + inputs[i+0];
      sum = sum * inputs[i+1];
      sum = sum - inputs[i+2];
      sum = sum / inputs[i+3];
    }
    // force evaluation & sanity check on numbers
    //std::cout << sum.toString() << std::endl;
    fl::eval(sum);
  };
  nonscalarSum();     // warmup
  time(nonscalarSum); // see comment at the top for numbers
}

//TEST(FooTest, binopFusionSharedInput) {
//  auto nonscalarSum = []() {
//    // same effect if we randomly initialize the data (constant folding won't be
//    // able to optimize this); making it all 1 so that result is more readable.
//    fl::Shape shape({10, 10});
//    std::vector<int> data(shape.elements(), 1);
//
//    auto iter = 10;
//    auto sum = fl::Tensor::fromVector(shape, data);
//    for (int i = 0; i < iter; i++) {
//      sum = sum + sum;
//    }
//    std::cout << sum.toString() << std::endl;
//  };
//
//  fl::setDefaultTensorType<OneDnnTensor>();
//  time(nonscalarSum);
//
//  fl::setDefaultTensorType<JitTensor<OneDnnTensor>>();
//  time(nonscalarSum);
//}
//
//
//TEST(FooTest, scalarNode) {
//  auto s = ScalarNode::create({2, 2}, dtype::b8, 20);
//  ASSERT_EQ(s->scalar<char>(), 20);
//  ASSERT_EQ(s->scalar<int>(), 20);
//
//  s = ScalarNode::create({2, 2}, dtype::f32, 1e30f);
//  ASSERT_EQ(s->scalar<float>(), 1e30f);
//  ASSERT_EQ(s->scalar<double>(), 1e30f);
//
//  s = ScalarNode::create({2, 2}, dtype::f32, 1e100);
//  ASSERT_EQ(s->scalar<double>(), 1e100);
//
//  s = ScalarNode::create({2, 2}, dtype::u32, 1 << 30);
//  ASSERT_EQ(s->scalar<int>(), 1 << 30);
//  ASSERT_EQ(s->scalar<unsigned int>(), 1 << 30);
//}
//
////TEST(FooTest, type) {
////  const auto C = 4;
////  const auto X = 4;
////  auto x = fl::full(Shape({C, X}), 0.0);
////  auto y = fl::full(Shape({1, X}), 1.0);
////
////  auto A = fl::arange(Shape({C, X}));
////  auto B = fl::tile(y, Shape({C}));
////  auto mask = -(A == B); // [C X]
////
////  auto type = mask.type();
////  std::cout << "mask: " << type << std::endl;
////
////  type = x.type();
////  std::cout << "x: " << type << std::endl;
////
////  // somehow `mask * x` dispatches to a b8 * b8 op in OneDnnBackend...
////  auto result = mask * x;
////  type = result.type();
////  std::cout << "result: " << type << std::endl;
////}
//
//TEST(FooTest, copyAndAssign) {
//  fl::setDefaultTensorType<JitTensor<OneDnnTensor>>();
//
//  // TODO need indexing for (=, assign)
//  fl::Shape shape({2, 2});
//  auto t0 = fl::full(shape, 0);
//  auto t1 = t0;
//  auto t2 = t0.shallowCopy();
//  std::cout << t2.toString() << std::endl; // 0
//
//  t0 += 1;
//  std::cout << t2.toString() << std::endl; // 1
//
//  std::cout << t0.toString() << std::endl; // 1
//  std::cout << t1.toString() << std::endl; // 0
//}
//
//TEST(FooTest, index) {
//  fl::setDefaultTensorType<JitTensor<ArrayFireTensor>>();
//
//  fl::Shape shape({2, 2});
//  auto t0 = fl::full({2}, 0);
//  auto t1 = fl::Tensor::fromVector<float>(shape, {0, 1, 2, 3});
//  auto t2 = t1(t0);
//  t1(0) += 10;
//  std::cout << t1.toString() << std::endl;
//  std::cout << t0.toString() << std::endl;
//  std::cout << t2.toString() << std::endl;
//}
//
//TEST(FooTest, pad) {
//  fl::setDefaultTensorType<ArrayFireTensor>();
//  auto t = fl::full({2, 2}, 0);
//  std::cout << fl::pad(t, {{1, 2}, {3, 4}}) << std::endl;
//}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
