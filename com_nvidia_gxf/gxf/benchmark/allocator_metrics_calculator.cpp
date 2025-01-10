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
#include "gxf/benchmark/allocator_metrics_calculator.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include "gxf/core/expected_macro.hpp"

namespace nvidia {
namespace gxf {
namespace benchmark {

// Compute Allocation and Free durations
gxf::Expected<void> ComputeBenchmarkAllocatorDuration(
    std::vector<std::chrono::nanoseconds>& timestamps_allocate_acqtime_ns,
    std::vector<std::chrono::nanoseconds>& timestamps_allocate_pubtime_ns,
    std::vector<std::chrono::nanoseconds>& timestamps_free_acqtime_ns,
    std::vector<std::chrono::nanoseconds>& timestamps_free_pubtime_ns,
    nlohmann::json& perf_results) {
  // Allocate duration based on acqtime and pubtime of receieved messages
  perf_results[AllocatorMetricsStrMap.at(AllocatorMetrics::kAllocateDuration)] =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          timestamps_allocate_pubtime_ns.back() - timestamps_allocate_acqtime_ns.front()).count();
  // Free duration based on acqtime and pubtime of receieved messages
  perf_results[AllocatorMetricsStrMap.at(AllocatorMetrics::kFreeDuration)] =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          timestamps_free_pubtime_ns.back() - timestamps_free_acqtime_ns.front()).count();
  return gxf::Success;
}

// Get statistics for given vectors of timestamps
gxf::Expected<MaxMinMeanSdValues<std::chrono::nanoseconds>> ComputeStatistics(
    const std::vector<std::chrono::nanoseconds>& timestamps_first_ns,
    const std::vector<std::chrono::nanoseconds>& timestamps_second_ns) {
  if (timestamps_first_ns.size() != timestamps_second_ns.size()) {
    GXF_LOG_ERROR(
        "The number of first and second timestamps vector were inconsistent"
        "for computing their mean delta");
    return gxf::Unexpected{GXF_FAILURE};
  }
  if (timestamps_first_ns.size() == 0) {
    GXF_LOG_ERROR("No timestamps were present for computing delta");
    return gxf::Unexpected{GXF_FAILURE};
  }

  std::vector<int64_t> diff_ns_list;
  for (size_t i = 0; i < timestamps_first_ns.size(); i++) {
    diff_ns_list.push_back(
        (timestamps_second_ns.at(i) - timestamps_first_ns.at(i)).count());
  }
  const auto results_int64 = UNWRAP_OR_RETURN(
      ComputeMaxMinMeanSdValues<int64_t>(diff_ns_list));

  MaxMinMeanSdValues<std::chrono::nanoseconds> results;
  results.max = static_cast<std::chrono::nanoseconds>(results_int64.max);
  results.min = static_cast<std::chrono::nanoseconds>(results_int64.min);
  results.mean = static_cast<std::chrono::nanoseconds>(results_int64.mean);
  results.sd = results_int64.sd;
  return results;
}

gxf::Expected<void> AllocatorMetricsCalculator::reset() {
  perf_history_.clear();
  return gxf::Success;
}

gxf::Expected<nlohmann::json> AllocatorMetricsCalculator::compute(
    std::vector<std::chrono::nanoseconds>& timestamps_allocate_acqtime_ns,
    std::vector<std::chrono::nanoseconds>& timestamps_allocate_pubtime_ns,
    std::vector<std::chrono::nanoseconds>& timestamps_free_acqtime_ns,
    std::vector<std::chrono::nanoseconds>& timestamps_free_pubtime_ns) {
  nlohmann::json perf_results;

  // Allocate/Free durations
  RETURN_IF_ERROR(
      ComputeBenchmarkAllocatorDuration(
          timestamps_allocate_acqtime_ns, timestamps_allocate_pubtime_ns,
          timestamps_free_acqtime_ns, timestamps_free_pubtime_ns, perf_results),
      "Failed to compute benchmark durations");

  // Statistics for allocate
  MaxMinMeanSdValues<std::chrono::nanoseconds> statistics_allcote =
      UNWRAP_OR_RETURN(ComputeStatistics(
          timestamps_allocate_acqtime_ns, timestamps_allocate_pubtime_ns));

  // Max allocate duration
  perf_results[AllocatorMetricsStrMap.at(AllocatorMetrics::kMaxAllocateDuration)] =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          statistics_allcote.max).count();

  // Min allocate duration
  perf_results[AllocatorMetricsStrMap.at(AllocatorMetrics::kMinAllocateDuration)] =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          statistics_allcote.min).count();

  // Mean allocate duration
  perf_results[AllocatorMetricsStrMap.at(AllocatorMetrics::kMeanAllocateDuration)] =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          statistics_allcote.mean).count();

  // Allocate standard deviation
  perf_results[AllocatorMetricsStrMap.at(AllocatorMetrics::kStdDevAllocateDuration)] =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          static_cast<std::chrono::duration<double, std::nano>>(
              statistics_allcote.sd)).count();

  // Statistics for free
  MaxMinMeanSdValues<std::chrono::nanoseconds> statistics_free =
      UNWRAP_OR_RETURN(ComputeStatistics(
          timestamps_free_acqtime_ns, timestamps_free_pubtime_ns));
  // Max free duration
  perf_results[AllocatorMetricsStrMap.at(AllocatorMetrics::kMaxFreeDuration)] =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          statistics_free.max).count();
  // Min free duration
  perf_results[AllocatorMetricsStrMap.at(AllocatorMetrics::kMinFreeDuration)] =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          statistics_free.min).count();
  // Mean free duration
  perf_results[AllocatorMetricsStrMap.at(AllocatorMetrics::kMeanFreeDuration)] =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          statistics_free.mean).count();
  // Free standard deviation
  perf_results[AllocatorMetricsStrMap.at(AllocatorMetrics::kStdDevFreeDuration)] =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          static_cast<std::chrono::duration<double, std::nano>>(
              statistics_free.sd)).count();

  perf_history_.push_back(perf_results);

  return GetNamespacedReport(perf_results, namespace_.get());
}

std::vector<double> AllocatorMetricsCalculator::getMetricValues(
    AllocatorMetrics metric_enum, bool should_filter = false) {
  std::vector<double> values;
  for (auto perf_result : perf_history_) {
    if (!perf_result.contains(AllocatorMetricsStrMap.at(metric_enum))) {
      return {};
    }
    values.push_back(perf_result[AllocatorMetricsStrMap.at(metric_enum)]);
  }

  if (should_filter) {
    if (values.size() <= 2) {
      GXF_LOG_WARNING(
          "Skipped filtering due to too few values (%ld)",
          values.size());
    } else {
      RemoveMinMaxValues(values);
    }
  }

  return values;
}

gxf::Expected<double> AllocatorMetricsCalculator::computeMetricMax(AllocatorMetrics metric) {
  auto values = getMetricValues(metric, true);
  if (values.size() == 0) {
    return gxf::Unexpected{GXF_QUERY_NOT_FOUND};
  }
  return *std::max_element(values.begin(), values.end());
}

gxf::Expected<double> AllocatorMetricsCalculator::computeMetricMin(AllocatorMetrics metric) {
  auto values = getMetricValues(metric, true);
  if (values.size() == 0) {
    return gxf::Unexpected{GXF_QUERY_NOT_FOUND};
  }
  return *std::min_element(values.begin(), values.end());
}

gxf::Expected<double> AllocatorMetricsCalculator::computeMetricMean(AllocatorMetrics metric) {
  auto values = getMetricValues(metric, true);
  if (values.size() == 0) {
    return gxf::Unexpected{GXF_QUERY_NOT_FOUND};
  }
  return UNWRAP_OR_RETURN(ComputeMean<double>(values));
}

nlohmann::json AllocatorMetricsCalculator::conclude() {
  nlohmann::json perf_results;

  for (auto const& [metric_enum, metric_str] : AllocatorMetricsStrMap) {
    switch (metric_enum) {
      case AllocatorMetrics::kMinAllocateDuration:
      case AllocatorMetrics::kMinFreeDuration:
        computeMetricMin(metric_enum)
            .map([&](double value) {
              perf_results[AllocatorMetricsStrMap.at(metric_enum)] = value;
            });
      case AllocatorMetrics::kMaxAllocateDuration:
      case AllocatorMetrics::kMaxFreeDuration:
        computeMetricMax(metric_enum)
            .map([&](double value) {
              perf_results[AllocatorMetricsStrMap.at(metric_enum)] = value;
            });
      case AllocatorMetrics::kAllocateDuration:
      case AllocatorMetrics::kFreeDuration:
      case AllocatorMetrics::kStdDevAllocateDuration:
      case AllocatorMetrics::kStdDevFreeDuration:
      case AllocatorMetrics::kMeanAllocateDuration:
      case AllocatorMetrics::kMeanFreeDuration:
        computeMetricMean(metric_enum)
            .map([&](double value) {
              perf_results[AllocatorMetricsStrMap.at(metric_enum)] = value;
            });
        break;
      default:
        break;
    }
  }

  return GetNamespacedReport(perf_results, namespace_.get());
}

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
