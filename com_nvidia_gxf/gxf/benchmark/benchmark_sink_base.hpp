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
#pragma once

#include <vector>

#include "gxf/benchmark/benchmark_publisher.hpp"
#include "gxf/benchmark/performance_calculator_base.hpp"
#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"

namespace nvidia {
namespace gxf {
namespace benchmark {

/**
 * @brief Base class for benchmark sink
 *
 * Benchmark sink is responsible for computing and providing all the statistics based on the type of
 * concrete class that would instantiated from the base class
 */
class BenchmarkSinkBase : public gxf::Codelet {
 public:
  // Signal the start of a benchmark iteration
  virtual gxf::Expected<void> begin() {
    return gxf::Success;
  }

  // Signal the end of a benchmark iteration
  virtual gxf::Expected<void> end() {
    return gxf::Success;
  }

  // Reset states of the benchmark sink and the associated perf calculators
  virtual gxf::Expected<void> reset() {
    return gxf::Success;
  }

  // Compute performance outcome.
  // The results are expected to be buffered in the associated perf calculators.
  virtual gxf::Expected<nlohmann::json> compute() = 0;
  // Conclude the performance results from the associated perf calculators
  virtual nlohmann::json conclude() = 0;

  // Getter of the recorded received timestamps
  virtual std::vector<std::chrono::nanoseconds>& getReceivedTimestamps() = 0;

  // Getter of the associated performance calculator component handles
  virtual gxf::Expected<std::vector<gxf::Handle<PerformanceCalculatorBase>>>
      getPerformanceCalculators() = 0;

  // Clear the runtime state
  // Calling this fucntion is sufficient to reset state for a new benchmark iteration
  virtual void clearRecordedTimestamps() {
    return;
  }
};

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
