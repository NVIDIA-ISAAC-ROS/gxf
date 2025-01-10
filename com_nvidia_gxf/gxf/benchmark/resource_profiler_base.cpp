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
#include <algorithm>
#include <vector>

#include "gxf/benchmark/resource_profiler_base.hpp"

namespace nvidia {
namespace gxf {
namespace benchmark {

gxf_result_t ResourceProfilerBase::registerInterface(gxf::Registrar* registrar) {
    gxf::Expected<void> result;
    auto parent_result = PerformanceCalculatorBase::registerInterface(registrar);
    if (parent_result != GXF_SUCCESS) {
      return parent_result;
    }
    result &= registerInterfaceHook(registrar);
    result &= registrar->parameter(
        profile_sampling_rate_hz_, "profile_sampling_rate_hz",
        "Resource profile sampling rate",
        "The sampling rate for profiling resource usage in Hz", 6.0);
    return gxf::ToResultCode(result);
  }

gxf_result_t ResourceProfilerBase::initialize() {
  return gxf::ToResultCode(initializeHook());
}

gxf_result_t ResourceProfilerBase::deinitialize() {
  if (is_profiling_) {
    GXF_LOG_INFO("Stopping resource profiling");
    is_profiling_ = false;
    profiling_thread_.join();
  }
  return gxf::ToResultCode(deinitializeHook());
}

gxf::Expected<void> ResourceProfilerBase::begin() {
  if (is_profiling_) {
    GXF_LOG_ERROR("Profiling was already active when trying to start profiling");
    return gxf::Unexpected{GXF_FAILURE};
  }

  overall_cpu_util_samples_.clear();
  host_memory_util_samples_.clear();
  device_util_samples_.clear();
  device_memory_util_samples_.clear();
  encoder_util_samples_.clear();
  decoder_util_samples_.clear();

  RETURN_IF_ERROR(beginHook());

  // Add a delay here so the next sampling is valid with a consistent time gap
  std::chrono::duration<double> duration_s(1.0/profile_sampling_rate_hz_.get());
  std::this_thread::sleep_for(duration_s);

  is_profiling_ = true;
  profiling_thread_ = std::thread([&]() {
    std::chrono::duration<double> duration_s(1.0/profile_sampling_rate_hz_.get());
    GXF_LOG_INFO("Begin resource profiling at %.3fHz (period=%.3fs)",
                 profile_sampling_rate_hz_.get(), duration_s.count());
    while (is_profiling_) {
      profileThreadHook();
      std::this_thread::sleep_for(duration_s);
    }
    GXF_LOG_INFO("Stopping profiling thread");
  });
  return gxf::Success;
}

gxf::Expected<void> ResourceProfilerBase::end() {
  if (is_profiling_ == false) {
    GXF_LOG_ERROR("Profiling was already inactive when trying to stop profiling");
    return gxf::Unexpected{GXF_FAILURE};
  }
  is_profiling_ = false;
  profiling_thread_.join();
  return endHook();
}

gxf::Expected<void> ResourceProfilerBase::reset() {
  overall_cpu_util_samples_.clear();
  host_memory_util_samples_.clear();
  device_util_samples_.clear();
  device_memory_util_samples_.clear();
  encoder_util_samples_.clear();
  decoder_util_samples_.clear();

  perf_history_.clear();
  return resetHook();
}

gxf::Expected<nlohmann::json> ResourceProfilerBase::compute() {
  nlohmann::json perf_results =
      UNWRAP_OR_RETURN(computeHook())[0];
  perf_history_.push_back(perf_results);
  return GetNamespacedReport(perf_results, namespace_.get());
}

nlohmann::json ResourceProfilerBase::conclude() {
  nlohmann::json perf_results;

  for (auto const& [metric_enum, metric_str] : kResourceMetricsStrMap) {
    switch (metric_enum) {
      case ResourceMetrics::kMaxOverallCPUUtilization:
      case ResourceMetrics::kMaxHostMemoryUtilization:
      case ResourceMetrics::kMaxDeviceUtilization:
      case ResourceMetrics::kMaxDeviceMemoryUtilization:
      case ResourceMetrics::kMaxEncoderUtilization:
      case ResourceMetrics::kMaxDecoderUtilization:
        computeMetricMax(metric_enum)
            .map([&](double value) {
              perf_results[kResourceMetricsStrMap.at(metric_enum)] = value;
            });
        break;
      case ResourceMetrics::kMinOverallCPUUtilization:
      case ResourceMetrics::kMinHostMemoryUtilization:
      case ResourceMetrics::kMinDeviceUtilization:
      case ResourceMetrics::kMinDeviceMemoryUtilization:
      case ResourceMetrics::kMinEncoderUtilization:
      case ResourceMetrics::kMinDecoderUtilization:
        computeMetricMin(metric_enum)
            .map([&](double value) {
              perf_results[kResourceMetricsStrMap.at(metric_enum)] = value;
            });
        break;
      case ResourceMetrics::kBaselineOverallCPUUtilization:
      case ResourceMetrics::kBaselineHostMemoryUtilization:
      case ResourceMetrics::kBaselineDeviceUtilization:
      case ResourceMetrics::kBaselineDeviceMemoryUtilization:
      case ResourceMetrics::kBaselineEncoderUtilization:
      case ResourceMetrics::kBaselineDecoderUtilization:
      case ResourceMetrics::kMeanOverallCPUUtilization:
      case ResourceMetrics::kMeanHostMemoryUtilization:
      case ResourceMetrics::kMeanDeviceUtilization:
      case ResourceMetrics::kMeanDeviceMemoryUtilization:
      case ResourceMetrics::kMeanEncoderUtilization:
      case ResourceMetrics::kMeanDecoderUtilization:
      case ResourceMetrics::kStdDevOverallCPUUtilization:
      case ResourceMetrics::kStdDevHostMemoryUtilization:
      case ResourceMetrics::kStdDevDeviceUtilization:
      case ResourceMetrics::kStdDevDeviceMemoryUtilization:
      case ResourceMetrics::kStdDevEncoderUtilization:
      case ResourceMetrics::kStdDevDecoderUtilization:
        computeMetricMean(metric_enum)
            .map([&](double value) {
              perf_results[kResourceMetricsStrMap.at(metric_enum)] = value;
            });
        break;
      default:
        break;
    }
  }

  return GetNamespacedReport(perf_results, namespace_.get());
}

gxf::Expected<double> ResourceProfilerBase::computeMetricMax(ResourceMetrics metric) {
  auto values = getMetricValues(metric, true);
  if (values.size() == 0) {
    return gxf::Unexpected{GXF_QUERY_NOT_FOUND};
  }
  return *std::max_element(values.begin(), values.end());
}

gxf::Expected<double> ResourceProfilerBase::computeMetricMin(ResourceMetrics metric) {
  auto values = getMetricValues(metric, true);
  if (values.size() == 0) {
    return gxf::Unexpected{GXF_QUERY_NOT_FOUND};
  }
  return *std::min_element(values.begin(), values.end());
}

gxf::Expected<double> ResourceProfilerBase::computeMetricMean(ResourceMetrics metric) {
  auto values = getMetricValues(metric, true);
  if (values.size() == 0) {
    return gxf::Unexpected{GXF_QUERY_NOT_FOUND};
  }
  return UNWRAP_OR_RETURN(ComputeMean<double>(values));
}

std::vector<double>
ResourceProfilerBase::getMetricValues(ResourceMetrics metric_enum, bool should_filter) {
  std::vector<double> values;
  for (auto perf_result : perf_history_) {
    if (!perf_result.contains(kResourceMetricsStrMap.at(metric_enum))) {
      return {};
    }
    values.push_back(perf_result[kResourceMetricsStrMap.at(metric_enum)]);
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

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
