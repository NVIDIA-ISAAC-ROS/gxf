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

#include <map>
#include <string>
#include <vector>

#include "gxf/benchmark/performance_calculator_base.hpp"

namespace nvidia {
namespace gxf {
namespace benchmark {

enum class AllocatorMetrics : uint8_t {
  kAllocateDuration,
  kMaxAllocateDuration,
  kMinAllocateDuration,
  kMeanAllocateDuration,
  kStdDevAllocateDuration,
  kFreeDuration,
  kMaxFreeDuration,
  kMinFreeDuration,
  kMeanFreeDuration,
  kStdDevFreeDuration
};

static const std::map<AllocatorMetrics, std::string> AllocatorMetricsStrMap = {
  {AllocatorMetrics::kAllocateDuration,
      "Total allocation duration based on pubtime and acqtime (ms)"},
  {AllocatorMetrics::kMaxAllocateDuration, "Max. Allocate duration (ms)"},
  {AllocatorMetrics::kMinAllocateDuration, "Min. Allcoate duration (ms)"},
  {AllocatorMetrics::kMeanAllocateDuration, "Mean Allocate duration (ms)"},
  {AllocatorMetrics::kStdDevAllocateDuration, "SD. Allocate duration (ms)"},
  {AllocatorMetrics::kFreeDuration,
      "Total free duration based on pubtime and acqtime (ms)"},
  {AllocatorMetrics::kMaxFreeDuration, "Max. Free duration (ms)"},
  {AllocatorMetrics::kMinFreeDuration, "Min. Free duration (ms)"},
  {AllocatorMetrics::kMeanFreeDuration, "Mean Free duration (ms)"},
  {AllocatorMetrics::kStdDevFreeDuration, "SD. Free duration (ms)"}
};

// A calculator for computing performance outcome of basic metrics
class AllocatorMetricsCalculator : public PerformanceCalculatorBase {
 public:
  // Reset all stored performance history
  gxf::Expected<void> reset() override;

  gxf::Expected<nlohmann::json> compute(
      std::vector<std::chrono::nanoseconds>& acqtime_ns,
      std::vector<std::chrono::nanoseconds>& pubtime_ns) {
      return nullptr;
  }

  // Compute performance results for allocator component
  gxf::Expected<nlohmann::json> compute(
    std::vector<std::chrono::nanoseconds>& timestamps_allocate_acqtime_ns,
    std::vector<std::chrono::nanoseconds>& timestamps_allocate_pubtime_ns,
    std::vector<std::chrono::nanoseconds>& timestamps_free_acqtime_ns,
    std::vector<std::chrono::nanoseconds>& timestamps_free_pubtime_ns) override;

  // Conclude performance results of basic metrics
  nlohmann::json conclude() override;

 private:
  // Get the max value for the specified mertic
  gxf::Expected<double> computeMetricMax(AllocatorMetrics metric);

  // Get the min value for the specified mertic
  gxf::Expected<double> computeMetricMin(AllocatorMetrics metric);

  // Compute the mean vaule for the specified metric
  gxf::Expected<double> computeMetricMean(AllocatorMetrics metric);

  // Get the stored computed values of the specified metric
  // Max and min values are excluded when should_filter is set to true
  std::vector<double> getMetricValues(AllocatorMetrics metric_enum, bool should_filter);

  std::vector<nlohmann::json> perf_history_;
};

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
