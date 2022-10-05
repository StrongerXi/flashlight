/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iomanip>
#include <iostream>

#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/common/Timer.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#include "flashlight/fl/tensor/backend/jit/JitTensor.h"
#include "flashlight/fl/tensor/backend/onednn/OneDnnTensor.h"

using namespace fl;

#define TIME_BACKEND(FUNC, TENSOR_BACKEND)                                                  \
  fl::setDefaultTensorType<TENSOR_BACKEND>();                                               \
  std::cout << "Timing " << #FUNC << " with " << #TENSOR_BACKEND << " ...  " << std::flush; \
  std::cout << std::setprecision(5) << FUNC() * 1000.0 << " msec" << std::endl;

#define TIME(FUNC)                            \
  TIME_BACKEND(FUNC, ArrayFireTensor)         \
  TIME_BACKEND(FUNC, OneDnnTensor)            \
  TIME_BACKEND(FUNC, JitTensor<OneDnnTensor>)

double timeit(std::function<void()> fn) {
  // warmup
  for (int i = 0; i < 0; ++i) {
    fn();
  }
  fl::sync();

  int iters = 1;
  fl::sync();
  auto start = fl::Timer::start();
  for (int i = 0; i < iters; i++) {
    fn();
  }
  fl::sync();
  return fl::Timer::stop(start) / iters;
}

double addMul() {
  Shape shape({1024, 1024});
  auto iters = 100;
  std::vector<Tensor> inputs;
  for (auto i = 0; i < 2 * iters + 1; i++) {
    inputs.push_back(fl::rand(shape, fl::dtype::f32));
    // only benchmark the arithmetic ops
    fl::eval(inputs.back());
  }

  auto fn = [&]() {
    auto res = inputs[0].copy();
    for (int i = 0; i < iters; i += 2) {
      res = res + inputs[i];
      res = res * inputs[i+1];
    }
    fl::eval(res);
  };
  return timeit(fn);
}

double reluMulForwardAndBackward() {
  Shape shape({1024, 1024});
  auto iters = 2;
  auto input = Variable(fl::rand(shape), true);
  auto target = max(input, 0.0);
  auto output = Variable(fl::rand(shape), true);
  fl::eval(input.tensor());
  fl::eval(target.tensor());
  fl::eval(output.tensor());

  auto fn = [&]() {
    for (int i = 0; i < iters; i++) {
      auto output = max(input, 0.0);
      output.backward();
      fl::eval(input.grad().tensor());
    }
  };
  return timeit(fn);
}

double matmul() {
  Shape shape({256, 256});
  auto iters = 10;
  auto lhs = fl::rand(shape, fl::dtype::f32);
  auto rhs = fl::rand(shape, fl::dtype::f32);

  auto fn = [&]() {
    for (int i = 0; i < iters; i++) {
      lhs = fl::matmul(lhs, rhs);
    }
    fl::eval(lhs);
  };
  return timeit(fn);
}

int main() {
  fl::init();
  //TIME(addMul);
  TIME(reluMulForwardAndBackward);
  //TIME(matmul);
  return 0;
}
