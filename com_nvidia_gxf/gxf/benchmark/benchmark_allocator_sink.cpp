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

#include "gxf/benchmark/benchmark_allocator.hpp"
#include "gxf/benchmark/benchmark_allocator_sink.hpp"
#include "gxf/core/expected_macro.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace gxf {
namespace benchmark {

gxf_result_t BenchmarkAllocatorSink::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(
      receiver_, "receiver",
      "Message sink receiver",
      "A receiver for retrieving incoming messages");
  result &= registrar->parameter(
      performance_calculators_,
      "performance_calculators",
      "Associated performance calculators",
      "A list of associated performance calculators for the incoming message flow",
      {});
  result &= registrar->parameter(
      benchmark_allocator_,
      "benchmark_allocator",
      "Benchmark Allocator",
      "Benchmark Allocator allocation and free calls",
      false);
  return gxf::ToResultCode(result);
}

gxf_result_t BenchmarkAllocatorSink::initialize() {
  is_benchmarking_ = true;

  return GXF_SUCCESS;
}

gxf_result_t BenchmarkAllocatorSink::tick() {
  auto message = receiver_->receive();
  if (!message) {
    GXF_LOG_ERROR("Failed to receive message: %d ", message.error());
    return message.error();
  }

  if (is_benchmarking_) {
    bool allocate{true};
    auto maybe_allocate = message.value().get<int32_t>("Allocate");
    if (!maybe_allocate) {
      // Probably message type is "Free"
      auto maybe_free = message.value().get<int32_t>("Free");
      if (!maybe_free) {
        GXF_LOG_ERROR("Unable to find either Allocate or Free in the message");
        return ToResultCode(maybe_allocate);
      }
      allocate = false;
    } else {
      allocate = true;
    }

    gxf::Handle<gxf::Timestamp> timestamp = UNWRAP_OR_RETURN(message->get<gxf::Timestamp>(),
        "Failed to get a timestamp component from the incoming message");
    if (allocate) {
      timestamps_allocate_acqtime_.push_back(std::chrono::nanoseconds(timestamp->acqtime));
      timestamps_allocate_pubtime_.push_back(std::chrono::nanoseconds(timestamp->pubtime));
      GXF_LOG_VERBOSE("Allocate acqtime = %ld pubtime = %ld",
                      timestamp->acqtime, timestamp->pubtime);
    } else {
      timestamps_free_acqtime_.push_back(std::chrono::nanoseconds(timestamp->acqtime));
      timestamps_free_pubtime_.push_back(std::chrono::nanoseconds(timestamp->pubtime));
      GXF_LOG_VERBOSE("Free acqtime = %ld pubtime = %ld",
                      timestamp->acqtime, timestamp->pubtime);
    }
  }
  return GXF_SUCCESS;
}

gxf::Expected<void> BenchmarkAllocatorSink::begin() {
  clearRecordedTimestamps();
  is_benchmarking_ = true;
  for (auto &perf_calculator : performance_calculators_.get()) {
    perf_calculator->begin();
  }
  return gxf::Success;
}

gxf::Expected<void> BenchmarkAllocatorSink::end() {
  is_benchmarking_ = false;
  for (auto &perf_calculator : performance_calculators_.get()) {
    perf_calculator->end();
  }
  return gxf::Success;
}

gxf::Expected<void> BenchmarkAllocatorSink::reset() {
  clearRecordedTimestamps();
  for (auto &perf_calculator : performance_calculators_.get()) {
    perf_calculator->reset();
  }
  return gxf::Success;
}

gxf::Expected<nlohmann::json> BenchmarkAllocatorSink::compute() {
  nlohmann::json perf_report;
  for (auto &perf_calculator : performance_calculators_.get()) {
    nlohmann::json this_perf_report =
        UNWRAP_OR_RETURN(perf_calculator->compute(
                            timestamps_allocate_acqtime_,
                            timestamps_allocate_pubtime_,
                            timestamps_free_acqtime_,
                            timestamps_free_pubtime_))[0];
    if (this_perf_report.is_object()) {
      perf_report.merge_patch(this_perf_report);
    }
  }

  return perf_report;
}

nlohmann::json BenchmarkAllocatorSink::conclude() {
  nlohmann::json perf_report;
  for (auto &perf_calculator : performance_calculators_.get()) {
    nlohmann::json this_perf_report = perf_calculator->conclude();
    if (this_perf_report.is_object()) {
      perf_report.merge_patch(this_perf_report);
    }
  }
  return perf_report;
}

std::vector<std::chrono::nanoseconds>& BenchmarkAllocatorSink::getReceivedTimestamps() {
  return timestamps_allocate_acqtime_;
}

gxf::Expected<std::vector<gxf::Handle<PerformanceCalculatorBase>>>
BenchmarkAllocatorSink::getPerformanceCalculators() {
  return performance_calculators_.try_get();
}

void BenchmarkAllocatorSink::clearRecordedTimestamps() {
  const size_t timestamp_count = timestamps_allocate_acqtime_.size();
  timestamps_allocate_acqtime_.clear();
  timestamps_allocate_pubtime_.clear();
  timestamps_free_acqtime_.clear();
  timestamps_free_pubtime_.clear();
  if (timestamp_count > 0) {
    timestamps_allocate_acqtime_.reserve(timestamp_count*1.5);
    timestamps_allocate_pubtime_.reserve(timestamp_count*1.5);
    timestamps_free_acqtime_.reserve(timestamp_count*1.5);
    timestamps_free_pubtime_.reserve(timestamp_count*1.5);
  }
}

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
