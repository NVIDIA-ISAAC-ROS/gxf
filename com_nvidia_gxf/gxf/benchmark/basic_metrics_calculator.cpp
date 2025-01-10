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
#include "gxf/benchmark/basic_metrics_calculator.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include "gxf/core/expected_macro.hpp"

namespace nvidia {
namespace gxf {
namespace benchmark {

// Compute latencies of the given timestamp pairs
gxf::Expected<MaxMinMeanSdValues<std::chrono::nanoseconds>> ComputeLatency(
    const std::vector<std::chrono::nanoseconds>& published_timestamps_ns,
    const std::vector<std::chrono::nanoseconds>& received_timestamps_ns) {
  std::chrono::nanoseconds latency_sum_ns{0}, max_latency_ns{0};
  std::chrono::nanoseconds min_latency_ns{INT64_MAX};
  if (published_timestamps_ns.size() != received_timestamps_ns.size()) {
    GXF_LOG_ERROR(
        "The number of published and received timestamps were inconsistent"
        "for computing their mean end-to-end latency");
    return gxf::Unexpected{GXF_FAILURE};
  }
  if (published_timestamps_ns.size() == 0) {
    GXF_LOG_ERROR("No timestamps were present for computing end-to-end latency");
    return gxf::Unexpected{GXF_FAILURE};
  }

  // Prepare raw latency (as int64_t) list for invoking ComputeMaxMinMeanSdValues()
  std::vector<int64_t> latency_ns_list;
  for (size_t i = 0; i < published_timestamps_ns.size(); i++) {
    latency_ns_list.push_back(
        (received_timestamps_ns.at(i) - published_timestamps_ns.at(i)).count());
  }
  const auto results_int64 = UNWRAP_OR_RETURN(
      ComputeMaxMinMeanSdValues<int64_t>(latency_ns_list));

  MaxMinMeanSdValues<std::chrono::nanoseconds> results;
  results.max = static_cast<std::chrono::nanoseconds>(results_int64.max);
  results.min = static_cast<std::chrono::nanoseconds>(results_int64.min);
  results.mean = static_cast<std::chrono::nanoseconds>(results_int64.mean);
  results.sd = results_int64.sd;
  return results;
}

// Compute frame latencies
gxf::Expected<void> ComputeBenchmarkLatencies(
const std::vector<std::chrono::nanoseconds>& published_timestamps_ns,
    const std::vector<std::chrono::nanoseconds>& received_timestamps_ns,
    nlohmann::json& perf_results) {
  const size_t published_count = published_timestamps_ns.size();
  const size_t received_count = received_timestamps_ns.size();
  // End-to-end frame latency
  if (published_count == received_count && published_count > 0) {
    MaxMinMeanSdValues<std::chrono::nanoseconds> e2e_latency =
        UNWRAP_OR_RETURN(ComputeLatency(
            published_timestamps_ns, received_timestamps_ns));
    // Max latency
    perf_results[BasicMetricsStrMap.at(BasicMetrics::kMaxEndToEndFrameLatency)] =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            e2e_latency.max).count();
    // Min latency
    perf_results[BasicMetricsStrMap.at(BasicMetrics::kMinEndToEndFrameLatency)] =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            e2e_latency.min).count();
    // Mean latency
    perf_results[BasicMetricsStrMap.at(BasicMetrics::kMeanEndToEndFrameLatency)] =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            e2e_latency.mean).count();
    // Latency standard deviation
    perf_results[BasicMetricsStrMap.at(BasicMetrics::kStdDevEndToEndFrameLatency)] =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            static_cast<std::chrono::duration<double, std::nano>>(
                e2e_latency.sd)).count();
  }

  if (received_count > 0 && published_count > 0) {
    // First sent to first received frame latency
    perf_results[BasicMetricsStrMap.at(BasicMetrics::kFirstSentReceivedLatency)] =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            received_timestamps_ns.at(0) - published_timestamps_ns.at(0)).count();

    // Last sent to last received frame latency
    const auto last_sent_received_latency_ns =
          received_timestamps_ns.at(received_count - 1) -
          published_timestamps_ns.at(published_count - 1);
    perf_results[BasicMetricsStrMap.at(BasicMetrics::kLastSentReceivedLatency)] =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            last_sent_received_latency_ns).count();
  }
  return gxf::Success;
}

// Compute jitters for the given timestamp series.
// Jitters are calculated as the absolute differences between two adjacent
// frame-to-fame durations.
gxf::Expected<MaxMinMeanSdValues<std::chrono::nanoseconds>> ComputeJitter(
    const std::vector<std::chrono::nanoseconds>& timestamps_ns) {
  if (timestamps_ns.size() < 2) {
    GXF_LOG_ERROR(
        "The number of given timestamps was insufficient for calculating jitters");
    return gxf::Unexpected{GXF_FAILURE};
  }

  // Prepare raw jitter (as int64_t) list for invoking ComputeMaxMinMeanSdValues()
  std::vector<std::chrono::nanoseconds> interarrival_ns_list;
  for (size_t i = 0; i < timestamps_ns.size() - 1; i++) {
    interarrival_ns_list.push_back(timestamps_ns.at(i + 1) - timestamps_ns.at(i));
  }
  std::vector<int64_t> jitter_ns_list;
  for (size_t i = 0; i < interarrival_ns_list.size() - 1; i++) {
    jitter_ns_list.push_back(
        abs((interarrival_ns_list.at(i + 1) - interarrival_ns_list.at(i)).count()));
  }
  const auto results_int64 = UNWRAP_OR_RETURN(
      ComputeMaxMinMeanSdValues<int64_t>(jitter_ns_list));

  MaxMinMeanSdValues<std::chrono::nanoseconds> results;
  results.max = static_cast<std::chrono::nanoseconds>(results_int64.max);
  results.min = static_cast<std::chrono::nanoseconds>(results_int64.min);
  results.mean = static_cast<std::chrono::nanoseconds>(results_int64.mean);
  results.sd = results_int64.sd;
  return results;
}


// Compuet jitters for received messages
gxf::Expected<void> ComputeBenchmarkJitters(
    const std::vector<std::chrono::nanoseconds>& received_timestamps_ns,
    nlohmann::json& perf_results) {
  if (received_timestamps_ns.size() > 2) {
    // Fram-to-frame (cycle-to-cycle) jitters
    MaxMinMeanSdValues<std::chrono::nanoseconds> jitter =
        UNWRAP_OR_RETURN(ComputeJitter(received_timestamps_ns));
    // Max jitter
    perf_results[BasicMetricsStrMap.at(BasicMetrics::kMaxFrameToFrameJitter)] =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            jitter.max).count();
    // Min jitter
    perf_results[BasicMetricsStrMap.at(BasicMetrics::kMinFrameToFrameJitter)] =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            jitter.min).count();
    // Mean jitter
    perf_results[BasicMetricsStrMap.at(BasicMetrics::kMeanFrameToFrameJitter)] =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            jitter.mean).count();
    // Jitter standard deviation
    perf_results[BasicMetricsStrMap.at(BasicMetrics::kStdDevFrameToFrameJitter)] =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            static_cast<std::chrono::duration<double, std::nano>>(
                jitter.sd)).count();
  }
  return gxf::Success;
}

// Compute the frequency of the given timestamps
gxf::Expected<double> ComputeFrequency(const std::vector<std::chrono::nanoseconds>& timestamps_ns) {
  if (timestamps_ns.size() < 2) {
    return 0.0;
  }
  auto min_max = std::minmax_element(timestamps_ns.begin(), timestamps_ns.end());
  auto duration_ns = *min_max.second - *min_max.first;
  return (timestamps_ns.size()-1)/std::chrono::duration_cast<std::chrono::duration<double>>(
    duration_ns).count();
}

// Compute published and received message frame rates
gxf::Expected<void> ComputeBenchmarkFrameRates(
const std::vector<std::chrono::nanoseconds>& published_timestamps_ns,
    const std::vector<std::chrono::nanoseconds>& received_timestamps_ns,
    nlohmann::json& perf_results) {
  // Publisher fps
  perf_results[BasicMetricsStrMap.at(BasicMetrics::kMeanPublisherFrameRate)] =
      UNWRAP_OR_RETURN(ComputeFrequency(published_timestamps_ns));
  // Output fps
  perf_results[BasicMetricsStrMap.at(BasicMetrics::kMeanOutputFrameRate)] =
      UNWRAP_OR_RETURN(ComputeFrequency(received_timestamps_ns));
  return gxf::Success;
}

// Get duration of the given timestamps
gxf::Expected<std::chrono::nanoseconds>
ComputeDuration(const std::vector<std::chrono::nanoseconds>& timestamps_ns) {
  auto min_max = std::minmax_element(timestamps_ns.begin(), timestamps_ns.end());
  return *min_max.second - *min_max.first;
}

// Compute published and received message durations
gxf::Expected<void> ComputeBenchmarkDurations(
    const std::vector<std::chrono::nanoseconds>& published_timestamps_ns,
    const std::vector<std::chrono::nanoseconds>& received_timestamps_ns,
    nlohmann::json& perf_results) {
  // Published duration
  perf_results[BasicMetricsStrMap.at(BasicMetrics::kSentDuration)] =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          UNWRAP_OR_RETURN(ComputeDuration(published_timestamps_ns))).count();
  // Received duration
  perf_results[BasicMetricsStrMap.at(BasicMetrics::kReceivedDuration)] =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          UNWRAP_OR_RETURN(ComputeDuration(received_timestamps_ns))).count();
  return gxf::Success;
}

gxf::Expected<void> BasicMetricsCalculator::reset() {
  perf_history_.clear();
  return gxf::Success;
}

gxf::Expected<nlohmann::json> BasicMetricsCalculator::compute(
    std::vector<std::chrono::nanoseconds>& published_timestamps_ns,
    std::vector<std::chrono::nanoseconds>& received_timestamps_ns) {
  nlohmann::json perf_results;
  const size_t published_count = published_timestamps_ns.size();
  const size_t received_count = received_timestamps_ns.size();

  // Sent & received count
  perf_results[BasicMetricsStrMap.at(BasicMetrics::kSentFrameCount)] =
      published_count;
  perf_results[BasicMetricsStrMap.at(BasicMetrics::kReceivedFrameCount)] =
      received_count;

  // Frame drop count
  if (published_count > 0) {
    perf_results[BasicMetricsStrMap.at(BasicMetrics::kMissedFrameCount)] =
        published_count - received_count;
  }

  // Sent and received durations
  RETURN_IF_ERROR(
      ComputeBenchmarkDurations(
          published_timestamps_ns, received_timestamps_ns, perf_results),
      "Failed to compute benchmark durations");

  // Sent and received frame rates
  RETURN_IF_ERROR(
      ComputeBenchmarkFrameRates(
          published_timestamps_ns, received_timestamps_ns, perf_results),
      "Failed to compute benchmark frame rates");

  // End-to-end, first-sent-first-received, last-sent-last-received lattencies
  RETURN_IF_ERROR(
      ComputeBenchmarkLatencies(
          published_timestamps_ns, received_timestamps_ns, perf_results),
      "Failed to compute frame latencies");

  // Received frame jitters
  RETURN_IF_ERROR(
      ComputeBenchmarkJitters(received_timestamps_ns, perf_results),
      "Failed to compute received message jitters");

  perf_history_.push_back(perf_results);

  return GetNamespacedReport(perf_results, namespace_.get());
}

std::vector<double> BasicMetricsCalculator::getMetricValues(
    BasicMetrics metric_enum, bool should_filter = false) {
  std::vector<double> values;
  for (auto perf_result : perf_history_) {
    if (!perf_result.contains(BasicMetricsStrMap.at(metric_enum))) {
      return {};
    }
    values.push_back(perf_result[BasicMetricsStrMap.at(metric_enum)]);
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

gxf::Expected<double> BasicMetricsCalculator::computeMetricMax(BasicMetrics metric) {
  auto values = getMetricValues(metric, true);
  if (values.size() == 0) {
    return gxf::Unexpected{GXF_QUERY_NOT_FOUND};
  }
  return *std::max_element(values.begin(), values.end());
}

gxf::Expected<double> BasicMetricsCalculator::computeMetricMin(BasicMetrics metric) {
  auto values = getMetricValues(metric, true);
  if (values.size() == 0) {
    return gxf::Unexpected{GXF_QUERY_NOT_FOUND};
  }
  return *std::min_element(values.begin(), values.end());
}

gxf::Expected<double> BasicMetricsCalculator::computeMetricMean(BasicMetrics metric) {
  auto values = getMetricValues(metric, true);
  if (values.size() == 0) {
    return gxf::Unexpected{GXF_QUERY_NOT_FOUND};
  }
  return UNWRAP_OR_RETURN(ComputeMean<double>(values));
}

nlohmann::json BasicMetricsCalculator::conclude() {
  nlohmann::json perf_results;

  for (auto const& [metric_enum, metric_str] : BasicMetricsStrMap) {
    switch (metric_enum) {
      case BasicMetrics::kMaxEndToEndFrameLatency:
      case BasicMetrics::kMaxFrameToFrameJitter:
        computeMetricMax(metric_enum)
            .map([&](double value) {
              perf_results[BasicMetricsStrMap.at(metric_enum)] = value;
            });
        break;
      case BasicMetrics::kMinEndToEndFrameLatency:
      case BasicMetrics::kMinFrameToFrameJitter:
        computeMetricMin(metric_enum)
            .map([&](double value) {
              perf_results[BasicMetricsStrMap.at(metric_enum)] = value;
            });
        break;
      case BasicMetrics::kSentDuration:
      case BasicMetrics::kReceivedDuration:
      case BasicMetrics::kMeanPublisherFrameRate:
      case BasicMetrics::kMeanOutputFrameRate:
      case BasicMetrics::kSentFrameCount:
      case BasicMetrics::kReceivedFrameCount:
      case BasicMetrics::kMissedFrameCount:
      case BasicMetrics::kFirstSentReceivedLatency:
      case BasicMetrics::kLastSentReceivedLatency:
      case BasicMetrics::kFirstFrameLatency:
      case BasicMetrics::kLastFrameLatency:
      case BasicMetrics::kMeanEndToEndFrameLatency:
      case BasicMetrics::kStdDevEndToEndFrameLatency:
      case BasicMetrics::kMeanFrameToFrameJitter:
      case BasicMetrics::kStdDevFrameToFrameJitter:
        computeMetricMean(metric_enum)
            .map([&](double value) {
              perf_results[BasicMetricsStrMap.at(metric_enum)] = value;
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
