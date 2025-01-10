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

#include <string>
#include <vector>

#include "gxf/core/component.hpp"
#include "nlohmann/json.hpp"

namespace nvidia {
namespace gxf {
namespace benchmark {

template <typename T>
struct MaxMinMeanSdValues {
  T max;
  T min;
  T mean;
  double sd;
};

// Base class of performance calculators
class PerformanceCalculatorBase : public gxf::Component {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) {
    gxf::Expected<void> result;
    result &= registrar->parameter(
        namespace_, "namespace",
        "Performance calculator namespace",
        "Namespace used to return performance results", std::string(""));
    return gxf::ToResultCode(result);
  }

  // Signal the start of a benchmark iteration
  virtual gxf::Expected<void> begin() {
    return gxf::Success;
  }

  // Signal the end of a benchmark iteration
  virtual gxf::Expected<void> end() {
    return gxf::Success;
  }

  // Reset calculator's state and clear any stored performance history
  virtual gxf::Expected<void> reset() {
    return gxf::Success;
  }

  // Compute performance outcome for the given timestamps and store the results
  virtual gxf::Expected<nlohmann::json> compute(
      std::vector<std::chrono::nanoseconds>& published_timestamps_ns,
      std::vector<std::chrono::nanoseconds>& received_timestamps_ns) = 0;

  // Compute performance outcome for the given pairs of timestamps and storre the results
  virtual gxf::Expected<nlohmann::json> compute(
      std::vector<std::chrono::nanoseconds>& timestamps_pair1_acqtime_ns,
      std::vector<std::chrono::nanoseconds>& timestamps_pair1_pubtime_ns,
      std::vector<std::chrono::nanoseconds>& timestamps_pair2_acqtime_ns,
      std::vector<std::chrono::nanoseconds>& timestamps_pair2_pubtime_ns) {
        GXF_LOG_WARNING("Not implemented");
    return nullptr;
  }

  // Conclude the performance results from the stored performance history
  virtual nlohmann::json conclude() = 0;

 protected:
  // Namespace used as a key for returning performance outcome in JSON
  gxf::Parameter<std::string> namespace_;
};

// Compute the mean value from the given vector
template <typename T>
gxf::Expected<double> ComputeMean(const std::vector<T> values) {
  if (values.size() == 0) {
    GXF_LOG_ERROR("No value was given to compute mean");
    return gxf::Unexpected{GXF_FAILURE};
  }
  double sum = 0.0;
  for (const auto& val : values) {
    sum += (T)val;
  }
  return sum/values.size();
}

// Compute the max, min, mean and standard deviation values from the given vector
template <typename T>
gxf::Expected<MaxMinMeanSdValues<T>> ComputeMaxMinMeanSdValues(
    const std::vector<T>& values) {
  // Calculate max, min and mean
  T value_sum{0}, max_value{0}, min_value{0};
  bool first{true};
  for (const auto & value : values) {
    if (first) {
      first = false;
      min_value = value;
    }
    value_sum += value;
    max_value = (value > max_value) ? value : max_value;
    min_value = (value < min_value) ? value : min_value;
  }
  const T mean_value = value_sum/values.size();

  // Calculate standard deviation
  double variance = 0.0;
  for (const auto & value : values) {
    const T abs_diff = value > mean_value ? value - mean_value : mean_value - value;
    variance += pow(abs_diff, 2);
  }
  const double sd_value{sqrt(variance/values.size())};

  MaxMinMeanSdValues<T> results;
  results.max = max_value;
  results.min = min_value;
  results.mean = mean_value;
  results.sd = sd_value;
  return results;
}

// Remove the max and min values from the given vector
template <typename T>
void RemoveMinMaxValues(std::vector<T> & values) {
  bool erase_second = false;
  auto min_max = std::minmax_element(values.begin(), values.end());
  if (min_max.second != min_max.first) {
    erase_second = true;
  }
  values.erase(min_max.first);
  if (erase_second) {
    if (min_max.second == values.end()) {
      values.erase(--min_max.second);
    } else {
      values.erase(min_max.second);
    }
  }
}

// Get namespaced string from a given report key
std::string GetNamespacedReportKey(
    const std::string& key,
    const std::string& prefix);

// Prepend a given prefix for each key in the given JSON report
nlohmann::json GetNamespacedReport(
    const nlohmann::json& report,
    const std::string& prefix);

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
