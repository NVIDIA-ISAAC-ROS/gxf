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
#include "gxf/benchmark/benchmark_sink_base.hpp"
#include "gxf/benchmark/performance_calculator_base.hpp"
#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"

namespace nvidia {
namespace gxf {
namespace benchmark {

// A benchmark sink that records message arrival timestamps
class BenchmarkAllocatorSink : public BenchmarkSinkBase {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t tick() override;

  // Signal the start of a benchmark iteration
  gxf::Expected<void> begin() override;

  // Signal the end of a benchmark iteration
  gxf::Expected<void> end() override;

  // Reset states of the benchmark sink and the associated perf calculators
  gxf::Expected<void> reset() override;

  // Compute performance outcome for the recorded timestamps.
  // The results are expected to be buffered in the associated perf calculators.
  gxf::Expected<nlohmann::json> compute() override;

  // Conclude the performance results from the associated perf calculators
  nlohmann::json conclude() override;

  // Getter of the recorded received timestamps
  std::vector<std::chrono::nanoseconds>& getReceivedTimestamps() override;

  // Getter of the associated performance calculator component handles
  gxf::Expected<std::vector<gxf::Handle<PerformanceCalculatorBase>>>
      getPerformanceCalculators() override;

  // Clear the runtime state
  // Calling this fucntion is sufficient to reset state for a new benchmark iteration
  void clearRecordedTimestamps() override;

 private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> receiver_;
  gxf::Parameter<std::vector<gxf::Handle<PerformanceCalculatorBase>>>
      performance_calculators_;
  gxf::Parameter<bool> benchmark_allocator_;

  bool is_benchmarking_{false};

  // acqtime and pubtime for allocater's allocate call
  std::vector<std::chrono::nanoseconds> timestamps_allocate_acqtime_;
  std::vector<std::chrono::nanoseconds> timestamps_allocate_pubtime_;

  // acqtime and pubtime for allocater's free call
  std::vector<std::chrono::nanoseconds> timestamps_free_acqtime_;
  std::vector<std::chrono::nanoseconds> timestamps_free_pubtime_;
};

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
