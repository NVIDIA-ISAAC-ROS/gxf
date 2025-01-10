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
#include <thread>
#include <vector>

#include "gxf/benchmark/performance_calculator_base.hpp"
#include "gxf/core/expected_macro.hpp"
#include "nlohmann/json.hpp"

namespace nvidia {
namespace gxf {
namespace benchmark {

enum class ResourceMetrics : uint8_t {
  kResourceProfilingSamplingRate,
  // CPU utilization
  kBaselineOverallCPUUtilization,
  kMaxOverallCPUUtilization,
  kMinOverallCPUUtilization,
  kMeanOverallCPUUtilization,
  kStdDevOverallCPUUtilization,
  // Host memory utilization
  kBaselineHostMemoryUtilization,
  kMaxHostMemoryUtilization,
  kMinHostMemoryUtilization,
  kMeanHostMemoryUtilization,
  kStdDevHostMemoryUtilization,
  // Device utilization
  kBaselineDeviceUtilization,
  kMaxDeviceUtilization,
  kMinDeviceUtilization,
  kMeanDeviceUtilization,
  kStdDevDeviceUtilization,
  // Device memory utilization
  kBaselineDeviceMemoryUtilization,
  kMaxDeviceMemoryUtilization,
  kMinDeviceMemoryUtilization,
  kMeanDeviceMemoryUtilization,
  kStdDevDeviceMemoryUtilization,
  // Encoder utilization
  kBaselineEncoderUtilization,
  kMaxEncoderUtilization,
  kMinEncoderUtilization,
  kMeanEncoderUtilization,
  kStdDevEncoderUtilization,
  // Decoder utilization
  kBaselineDecoderUtilization,
  kMaxDecoderUtilization,
  kMinDecoderUtilization,
  kMeanDecoderUtilization,
  kStdDevDecoderUtilization
};

static const std::map<ResourceMetrics, std::string> kResourceMetricsStrMap = {
  {ResourceMetrics::kResourceProfilingSamplingRate, "Device Profile Sampling Rate (Hz)"},
  // CPU utilization
  {ResourceMetrics::kBaselineOverallCPUUtilization, "Baseline Overall CPU Utilization (%)"},
  {ResourceMetrics::kMaxOverallCPUUtilization, "Max. Overall CPU Utilization (%)"},
  {ResourceMetrics::kMinOverallCPUUtilization, "Min. Overall CPU Utilization (%)"},
  {ResourceMetrics::kMeanOverallCPUUtilization, "Mean Overall CPU Utilization (%)"},
  {ResourceMetrics::kStdDevOverallCPUUtilization, "Std Dev Overall CPU Utilization (%)"},
  // Host memory utilization
  {ResourceMetrics::kBaselineHostMemoryUtilization, "Baseline Host Memory Utilization (%)"},
  {ResourceMetrics::kMaxHostMemoryUtilization, "Max. Host Memory Utilization (%)"},
  {ResourceMetrics::kMinHostMemoryUtilization, "Min. Host Memory Utilization (%)"},
  {ResourceMetrics::kMeanHostMemoryUtilization, "Mean Host Memory Utilization (%)"},
  {ResourceMetrics::kStdDevHostMemoryUtilization, "SD. Host Memory Utilization (%)"},
  // Device utilization
  {ResourceMetrics::kBaselineDeviceUtilization, "Baseline Device Utilization (%)"},
  {ResourceMetrics::kMaxDeviceUtilization, "Max. Device Utilization (%)"},
  {ResourceMetrics::kMinDeviceUtilization, "Min. Device Utilization (%)"},
  {ResourceMetrics::kMeanDeviceUtilization, "Mean Device Utilization (%)"},
  {ResourceMetrics::kStdDevDeviceUtilization, "SD. Device Utilization (%)"},
  // Device memory utilization
  {ResourceMetrics::kBaselineDeviceMemoryUtilization, "Baseline Device Memory Utilization (%)"},
  {ResourceMetrics::kMaxDeviceMemoryUtilization, "Max. Device Memory Utilization (%)"},
  {ResourceMetrics::kMinDeviceMemoryUtilization, "Min. Device Memory Utilization (%)"},
  {ResourceMetrics::kMeanDeviceMemoryUtilization, "Mean Device Memory Utilization (%)"},
  {ResourceMetrics::kStdDevDeviceMemoryUtilization, "SD. Device Memory Utilization (%)"},
  // Encoder utilization
  {ResourceMetrics::kBaselineEncoderUtilization, "Baseline Encoder Utilization (%)"},
  {ResourceMetrics::kMaxEncoderUtilization, "Max. Encoder Utilization (%)"},
  {ResourceMetrics::kMinEncoderUtilization, "Min. Encoder Utilization (%)"},
  {ResourceMetrics::kMeanEncoderUtilization, "Mean Encoder Utilization (%)"},
  {ResourceMetrics::kStdDevEncoderUtilization, "SD. Encoder Utilization (%)"},
  // Decoder utilization
  {ResourceMetrics::kBaselineDecoderUtilization, "Baseline Decoder Utilization (%)"},
  {ResourceMetrics::kMaxDecoderUtilization, "Max. Decoder Utilization (%)"},
  {ResourceMetrics::kMinDecoderUtilization, "Min. Decoder Utilization (%)"},
  {ResourceMetrics::kMeanDecoderUtilization, "Mean Decoder Utilization (%)"},
  {ResourceMetrics::kStdDevDecoderUtilization, "SD. Decoder Utilization (%)"},
};

// A profiler for measuring resource usage
class ResourceProfilerBase : public PerformanceCalculatorBase {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  virtual gxf::Expected<void> registerInterfaceHook(gxf::Registrar* registrar) {
    return gxf::Success;
  }

  // Initialize resource profiler
  gxf_result_t initialize() override;
  virtual gxf::Expected<void> initializeHook() {
    return gxf::Success;
  }

  // Deinitialize resource profiler
  gxf_result_t deinitialize() override;
  virtual gxf::Expected<void> deinitializeHook() {
    return gxf::Success;
  }

  // Start resource profiling thread
  gxf::Expected<void> begin() override;
  virtual gxf::Expected<void> beginHook() {
    return gxf::Success;
  }

  // Stop resource profiling thread
  gxf::Expected<void> end() override;
  virtual gxf::Expected<void> endHook() {
    return gxf::Success;
  }

  // Hook function called in resource profiling thread
  virtual gxf::Expected<void> profileThreadHook() {
    return gxf::Success;
  }

  // Reset all stored profiling history
  gxf::Expected<void> reset() override;
  virtual gxf::Expected<void> resetHook() {
    return gxf::Success;
  }

  gxf::Expected<nlohmann::json> compute(
      std::vector<std::chrono::nanoseconds>& published_timestamps_ns,
      std::vector<std::chrono::nanoseconds>& received_timestamps_ns) override {
    return nlohmann::json{};
  };

  // Compute profiling results for a benchmark iteration
  gxf::Expected<nlohmann::json> compute();
  virtual gxf::Expected<nlohmann::json> computeHook() {
    return nlohmann::json{};
  }

  // Conclude profiling results
  nlohmann::json conclude() override;

 protected:
  // Get the max value for the specified mertic
  gxf::Expected<double> computeMetricMax(ResourceMetrics metric);

  // Get the min value for the specified mertic
  gxf::Expected<double> computeMetricMin(ResourceMetrics metric);

  // Compute the mean vaule for the specified metric
  gxf::Expected<double> computeMetricMean(ResourceMetrics metric);

  // Get the stored computed values of the specified metric
  // Max and min values are excluded when should_filter is set to true
  std::vector<double> getMetricValues(ResourceMetrics metric_enum, bool should_filter);

  // Parameters
  gxf::Parameter<double> profile_sampling_rate_hz_;

  std::vector<nlohmann::json> perf_history_;
  std::thread profiling_thread_;
  bool is_profiling_;

  // Profiling variables used within a benchmark iteration
  double baseline_overall_cpu_util_;
  double baseline_host_memory_util_;
  double baseline_device_util_;
  double baseline_device_memory_util_;
  double baseline_encoder_util_;
  double baseline_decoder_util_;
  std::vector<double> overall_cpu_util_samples_;
  std::vector<double> host_memory_util_samples_;
  std::vector<double> device_util_samples_;
  std::vector<double> device_memory_util_samples_;
  std::vector<double> encoder_util_samples_;
  std::vector<double> decoder_util_samples_;
};

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
