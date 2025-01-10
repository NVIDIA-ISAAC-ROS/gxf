/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef NVIDIA_GXF_BENCHMARK_BENCHMARK_ALLOCATOR_HPP_
#define NVIDIA_GXF_BENCHMARK_BENCHMARK_ALLOCATOR_HPP_

#include <vector>

#include "gxf/core/expected_macro.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/clock.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {
namespace benchmark {

/**
 * @brief Enum for determining the allocation type that is being profiled
 *
 */
enum class AllocationType : int32_t {
  kAllocate = 0,
  kFree,
};

class BenchmarkAllocator : public Codelet {
 public:
  BenchmarkAllocator() = default;
  ~BenchmarkAllocator();

  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t tick() override;

 private:
  Parameter<int32_t> number_of_iterations_;
  Parameter<int32_t> storage_type_;
  Parameter<int32_t> number_of_blocks_;
  Parameter<uint64_t> block_size_;
  Parameter<Handle<Allocator>> allocator_;
  Parameter<Handle<Clock>> clock_;
  Parameter<Handle<Transmitter>> output_;
  Parameter<Handle<Receiver>> in_;

  std::vector<nvidia::byte*> pointer_;
  int32_t allocation_count_{0};
  int32_t iteration_count_{0};
  AllocationType allocation_type_{AllocationType::kAllocate};
};

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_BENCHMARK_BENCHMARK_ALLOCATOR_HPP_
