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

enum class BasicMetrics : uint8_t {
  kSentDuration,
  kReceivedDuration,
  kMeanPublisherFrameRate,
  kMeanOutputFrameRate,
  kSentFrameCount,
  kReceivedFrameCount,
  kMissedFrameCount,
  kFirstSentReceivedLatency,
  kLastSentReceivedLatency,
  kFirstFrameLatency,
  kLastFrameLatency,
  kMaxEndToEndFrameLatency,
  kMinEndToEndFrameLatency,
  kMeanEndToEndFrameLatency,
  kStdDevEndToEndFrameLatency,
  kMaxFrameToFrameJitter,
  kMinFrameToFrameJitter,
  kMeanFrameToFrameJitter,
  kStdDevFrameToFrameJitter
};

static const std::map<BasicMetrics, std::string> BasicMetricsStrMap = {
  {BasicMetrics::kSentDuration, "Delta between First & Last Sent Frames (ms)"},
  {BasicMetrics::kReceivedDuration, "Delta between First & Last Received Frames (ms)"},
  {BasicMetrics::kMeanPublisherFrameRate, "Mean Publisher Frame Rate (fps)"},
  {BasicMetrics::kMeanOutputFrameRate, "Mean Frame Rate (fps)"},
  {BasicMetrics::kSentFrameCount, "# of Frames Sent"},
  {BasicMetrics::kReceivedFrameCount, "# of Frames Received"},
  {BasicMetrics::kMissedFrameCount, "# of Missed Frames"},
  {BasicMetrics::kFirstSentReceivedLatency, "First Sent to First Received Latency (ms)"},
  {BasicMetrics::kLastSentReceivedLatency, "Last Sent to Last Received Latency (ms)"},
  {BasicMetrics::kFirstFrameLatency, "First Frame End-to-end Latency (ms)"},
  {BasicMetrics::kLastFrameLatency, "Last Frame End-to-end Latency (ms)"},
  {BasicMetrics::kMaxEndToEndFrameLatency, "Max. End-to-End Latency (ms)"},
  {BasicMetrics::kMinEndToEndFrameLatency, "Min. End-to-End Latency (ms)"},
  {BasicMetrics::kMeanEndToEndFrameLatency, "Mean End-to-End Latency (ms)"},
  {BasicMetrics::kStdDevEndToEndFrameLatency, "SD. End-to-End Latency (ms)"},
  {BasicMetrics::kMaxFrameToFrameJitter, "Max. Frame-to-Frame Jitter (ms)"},
  {BasicMetrics::kMinFrameToFrameJitter, "Min. Frame-to-Frame Jitter (ms)"},
  {BasicMetrics::kMeanFrameToFrameJitter, "Mean Frame-to-Frame Jitter (ms)"},
  {BasicMetrics::kStdDevFrameToFrameJitter, "SD. Frame-to-Frame Jitter (ms)"}
};

// A calculator for computing performance outcome of basic metrics
class BasicMetricsCalculator : public PerformanceCalculatorBase {
 public:
  // Reset all stored performance history
  gxf::Expected<void> reset() override;

  // Compute performance results for basic metrics
  gxf::Expected<nlohmann::json> compute(
      std::vector<std::chrono::nanoseconds>& published_timestamps_ns,
      std::vector<std::chrono::nanoseconds>& received_timestamps_ns) override;

  // Conclude performance results of basic metrics
  nlohmann::json conclude() override;

 private:
  // Get the max value for the specified mertic
  gxf::Expected<double> computeMetricMax(BasicMetrics metric);

  // Get the min value for the specified mertic
  gxf::Expected<double> computeMetricMin(BasicMetrics metric);

  // Compute the mean vaule for the specified metric
  gxf::Expected<double> computeMetricMean(BasicMetrics metric);

  // Get the stored computed values of the specified metric
  // Max and min values are excluded when should_filter is set to true
  std::vector<double> getMetricValues(BasicMetrics metric_enum, bool should_filter);

  std::vector<nlohmann::json> perf_history_;
};

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
