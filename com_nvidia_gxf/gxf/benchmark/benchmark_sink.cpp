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
#include "gxf/benchmark/benchmark_sink.hpp"

#include <algorithm>
#include <vector>

#include "gxf/core/expected_macro.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace gxf {
namespace benchmark {

gxf_result_t BenchmarkSink::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(
      receiver_, "receiver",
      "Message sink receiver",
      "A receiver for retrieving incoming messages");
  result &= registrar->parameter(
      benchmark_publisher_, "benchmark_publisher",
      "A data source benchmark publisher",
      "A benchmark publisher for retrieving published timestamps",
      gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      performance_calculators_,
      "performance_calculators",
      "Associated performance calculators",
      "A list of associated performance calculators for the incoming message flow",
      {});
  result &= registrar->parameter(
      use_received_acqtime_as_published_,
      "use_received_acqtime_as_published",
      "Use acqtime from incoming messages",
      "Use acqtime from incoming messages as published message timestamps",
      false);
  return gxf::ToResultCode(result);
}

gxf_result_t BenchmarkSink::initialize() {
  is_benchmarking_ = false;
  if (use_received_acqtime_as_published_.get() && benchmark_publisher_.try_get()) {
    GXF_LOG_WARNING(
        "Parameter \"benchmark_publisher\" was ignored as \""
        "use_received_acqtime_as_published\" was set to true");
  }
  return GXF_SUCCESS;
}

gxf_result_t BenchmarkSink::tick() {
  auto message = receiver_->receive();
  if (!message) {
    return message.error();
  }
  if (is_benchmarking_) {
    received_timestamps_.push_back(
        std::chrono::nanoseconds(getExecutionTimestamp()));

    // Record acqtime from the received message if needed
    if (use_received_acqtime_as_published_.get()) {
      gxf::Handle<gxf::Timestamp> timestamp =
          UNWRAP_OR_RETURN(message->get<gxf::Timestamp>(),
          "Failed to get a timestamp component from the incoming message");
      published_acqtimes_.push_back(
          std::chrono::nanoseconds(timestamp->acqtime));
    }
  }
  return GXF_SUCCESS;
}

gxf::Expected<void> BenchmarkSink::begin() {
  clearRecordedTimestamps();
  is_benchmarking_ = true;
  for (auto &perf_calculator : performance_calculators_.get()) {
    perf_calculator->begin();
  }
  return gxf::Success;
}

gxf::Expected<void> BenchmarkSink::end() {
  is_benchmarking_ = false;
  for (auto &perf_calculator : performance_calculators_.get()) {
    perf_calculator->end();
  }
  return gxf::Success;
}

gxf::Expected<void> BenchmarkSink::reset() {
  clearRecordedTimestamps();
  for (auto &perf_calculator : performance_calculators_.get()) {
    perf_calculator->reset();
  }
  return gxf::Success;
}

gxf::Expected<nlohmann::json> BenchmarkSink::compute() {
  nlohmann::json perf_report;
  std::vector<std::chrono::nanoseconds> published_timestamps;
  if (use_received_acqtime_as_published_.get()) {
    published_timestamps = published_acqtimes_;
  } else if (benchmark_publisher_.try_get()) {
    published_timestamps = benchmark_publisher_.try_get().value()->getPublishedTimestamps();
  }
  for (auto &perf_calculator : performance_calculators_.get()) {
    nlohmann::json this_perf_report =
        UNWRAP_OR_RETURN(perf_calculator->compute(
                         published_timestamps,
                         received_timestamps_))[0];
      if (this_perf_report.is_object()) {
        perf_report.merge_patch(this_perf_report);
      }
  }
  return perf_report;
}

nlohmann::json BenchmarkSink::conclude() {
  nlohmann::json perf_report;
  for (auto &perf_calculator : performance_calculators_.get()) {
    nlohmann::json this_perf_report = perf_calculator->conclude();
    if (this_perf_report.is_object()) {
      perf_report.merge_patch(this_perf_report);
    }
  }
  return perf_report;
}

std::vector<std::chrono::nanoseconds>& BenchmarkSink::getReceivedTimestamps() {
  return received_timestamps_;
}

gxf::Expected<std::vector<gxf::Handle<PerformanceCalculatorBase>>>
BenchmarkSink::getPerformanceCalculators() {
  return performance_calculators_.try_get();
}

void BenchmarkSink::clearRecordedTimestamps() {
  const size_t timestamp_count =
      std::max(received_timestamps_.size(), published_acqtimes_.size());
  received_timestamps_.clear();
  published_acqtimes_.clear();
  if (timestamp_count > 0) {
    received_timestamps_.reserve(timestamp_count*1.5);
    published_acqtimes_.reserve(timestamp_count*1.5);
  }
}

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
