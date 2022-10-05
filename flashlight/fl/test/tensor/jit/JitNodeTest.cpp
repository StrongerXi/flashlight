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
  using micro = std::chrono::microseconds;
  auto start = std::chrono::high_resolution_clock::now();

  func();

  auto finish = std::chrono::high_resolution_clock::now();
  std::cout << "Took "
            << (std::chrono::duration_cast<micro>(finish - start).count() / 1000.0)
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

  // easy to enable JIT for any backend
  //fl::setDefaultTensorType<OneDnnTensor>();            // ~300ms
  fl::setDefaultTensorType<JitTensor<OneDnnTensor>>(); // ~15ms
  fl::Shape shape({1000, 1000});
  auto iters = 30;
  // data init
  std::vector<fl::Tensor> inputs;
  for (int i = 0; i < iters; i++) {
    //std::vector<int> data;
    //for (int j = 0; j < shape.elements(); j++) {
    //  data.push_back(i + j);
    //}
    //inputs.push_back(fl::Tensor::fromVector(shape, data));
    inputs.push_back(fl::rand(shape));
    fl::eval(inputs.back());
  }
  // benchmark binops only, not input data allocation
  auto nonscalarSum = [&]() {
    auto sum = inputs[0];
    for (int i = 1; i < iters; i++) {
      sum = sum + inputs[i];
    }
    // force evaluation & sanity check on numbers
    //std::cout << sum.toString() << std::endl;
    fl::eval(sum);
  };
  nonscalarSum();     // warmup
  time(nonscalarSum); // see comment at the top for numbers
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
