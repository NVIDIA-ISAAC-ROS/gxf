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

#include <memory>
#include <string>
#include <vector>

#include "gxf/benchmark/benchmark_allocator_sink.hpp"
#include "gxf/benchmark/benchmark_publisher.hpp"
#include "gxf/benchmark/benchmark_sink.hpp"
#include "gxf/benchmark/entity_buffer.hpp"
#include "gxf/benchmark/gems/data_replay_control.hpp"
#include "gxf/benchmark/resource_profiler_base.hpp"
#include "gxf/core/expected_macro.hpp"
#include "gxf/serialization/file.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/scheduling_terms.hpp"

namespace nvidia {
namespace gxf {
namespace benchmark {

// GXF benchmark controller state
enum BenchmarkState {
  kBenchmarkStateStopped = 0,
  kBenchmarkStateInit = 1,
  kBenchmarkStateWaitingDependencies,
  kBenchmarkStateStartBuffering,
  kBenchmarkStateBuffering,
  kBenchmarkStateStopBuffering,
  kBenchmarkStateSetUpTestCase,
  kBenchmarkStateStartRunningTestIteration,
  kBenchmarkStateRunningTestIteration,
  kBenchmarkStateSummarizeTestIteration,
  kBenchmarkStateConcludeTestCase,
  kBenchmarkStateExportReport,
  kBenchmarkStateEnding,
  kBenchmarkStateTerminating,
  kBenchmarkStateUndefined
};

// A benchmark controller that governs the entire benchmark flow
class BenchmarkController : public gxf::Codelet {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t start() override;
  gxf_result_t tick() override;

 private:
  // Handle BenchmarkStateInit for setting up initial states for benchmarking
  gxf::Expected<BenchmarkState> handleBenchmarkStateInit();

  // Handle BenchmarkStateWaitingDependencies to wait for dependent components to be ready
  gxf::Expected<BenchmarkState> handleBenchmarkStateWaitingDependencies();

  // Handle BenchmarkStateStartBuffering for signaling to start buffering
  gxf::Expected<BenchmarkState> handleBenchmarkStateStartBuffering();

  // Handle BenchmarkStateBuffering for buffering benchmark messages
  gxf::Expected<BenchmarkState> handleBenchmarkStateBuffering();

  // Handle BenchmarkStateStopBuffering for signaling to stop buffering
  gxf::Expected<BenchmarkState> handleBenchmarkStateStopBuffering();

  // Handle BenchmarkStateSetUpTestCase for preparing for a new benchmark test case
  gxf::Expected<BenchmarkState> handleBenchmarkStateSetUpTestCase();

  // Handle BenchmarkStateStartRunningTestIteration for starting a new benchmark iteration
  gxf::Expected<BenchmarkState> handleBenchmarkStateStartRunningTestIteration();

  // Handle BenchmarkStateRunningTestIteration for running a benchmark iteration
  gxf::Expected<BenchmarkState> handleBenchmarkStateRunningTestIteration();

  // Handle BenchmarkStateSummarizeTestIteration for summarizing a benchmark iteration
  gxf::Expected<BenchmarkState> handleBenchmarkStateSummarizeTestIteration();

  // Handle BenchmarkStateConcludeTestCase for concluding the currnet test case
  gxf::Expected<BenchmarkState> handleBenchmarkStateConcludeTestCase();

  // Handle BenchmarkStateExportReport for exporting the final report
  gxf::Expected<BenchmarkState> handleBenchmarkStateExportReport();

  // Handle BenchmarkStateEnding for ending the benchmark controller
  gxf::Expected<BenchmarkState> handleBenchmarkStateEnding();

  // Handle BenchmarkStateTerminating for terminating the app due to flow failures
  void handleBenchmarkStateTerminating();

  // Compute performance outcome and generate report for the current benchmark iteration
  gxf::Expected<nlohmann::json> summarizeCurrentTestIteration();

  // Check if all entity buffers from all associated benchmark publishers are fully loaded
  bool areAllBenchmarkPublisherBuffersFull() const;

  // Signal to begin benchmarking
  gxf::Expected<void> signalBeginBenchmarking();

  // Signal to end benchmarking for resource profilers
  gxf::Expected<void> signalEndBenchmarkingToResourceProfilers();

  // Signal to end benchmarking for benchmark sinks
  gxf::Expected<void> signalEndBenchmarkingToBenchmarkSinks();

  // Reset all buffered information and states in all assocated resource profilers
  gxf::Expected<void> resetResourceProfilers();

  // Reset all buffered information and states in all assocated benchmark sinks
  gxf::Expected<void> resetBenchmarkSinks();

  // Clear all buffered timestamp information in all associated benchmark publishers
  void clearBenchmarkPublishers();

  // Clear all buffered information in all associated benchmark sinks
  void clearBenchmarkSinks();

  // Stop all benchmark publishers
  void stopBenchmarkPublishers();

  // Get final report for the currently running benchmark test case
  gxf::Expected<nlohmann::json> concludeTestCase();

  // Print the given performance report by using GXF_LOG_INFO in a formatted way
  void printReport(nlohmann::json report, std::string sub_heading);

  // Send a control command to the connected data replayer
  gxf::Expected<void> sendDataReplayCommand(
      nvidia::gxf::benchmark::DataReplayControl::Command command);

  // Pause the connected data replayer
  gxf::Expected<void> pauseDataReplay();

  // Signal the connected data replayer to play
  gxf::Expected<void> playDataReplay();

  // Handles of scheduling terms to control benchmark flow
  gxf::Parameter<std::vector<gxf::Handle<gxf::BooleanSchedulingTerm>>>
      data_source_boolean_scheduling_terms_;
  gxf::Parameter<std::vector<gxf::Handle<gxf::AsynchronousSchedulingTerm>>>
      data_source_async_scheduling_terms_;
  gxf::Parameter<gxf::Handle<gxf::BooleanSchedulingTerm>>
      benchmark_controller_boolean_scheduling_term_;
  gxf::Parameter<gxf::Handle<gxf::TargetTimeSchedulingTerm>>
      benchmark_controller_target_time_scheduling_term_;
  gxf::Parameter<std::vector<gxf::Handle<gxf::BooleanSchedulingTerm>>>
      graph_boolean_scheduling_terms_;

  // Handles of components to wait for their GXF states to be ready
  gxf::Parameter<std::vector<gxf::Handle<gxf::Component>>> dependent_components_;

  // Transmitter to control the connected data replayer
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> data_replay_control_transmitter_;

  // Benchmark publishers and sinks
  gxf::Parameter<std::vector<gxf::Handle<BenchmarkPublisher>>> benchmark_publishers_;
  gxf::Parameter<std::vector<gxf::Handle<BenchmarkSinkBase>>> benchmark_sinks_;

  // Resource Profilers
  gxf::Parameter<std::vector<gxf::Handle<ResourceProfilerBase>>> resource_profilers_;

  // Benchmark configurations
  gxf::Parameter<bool> trial_run_;
  gxf::Parameter<bool> kill_at_the_end_;
  gxf::Parameter<std::string> title_;
  gxf::Parameter<gxf::Handle<gxf::File>> exported_report_;
  gxf::Parameter<size_t> entity_buffer_size_;
  gxf::Parameter<uint64_t> benchmark_duration_ms_;
  gxf::Parameter<size_t> benchmark_iterations_;
  gxf::Parameter<uint64_t> benchmark_buffering_timeout_s_;
  gxf::Parameter<uint64_t> post_trial_benchmark_iteration_delay_s_;
  gxf::Parameter<uint64_t> post_benchmark_iteration_delay_s_;

  // Internal benchmark states
  BenchmarkState benchmark_state_;
  BenchmarkState last_benchmark_state_;
  int64_t next_timeout_timestamp_ns_;
  size_t benchmark_iteration_count_;
  bool is_trial_running_;
  nlohmann::json report_;
};

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia

namespace nvidia::expected_macro {

template <>
constexpr nvidia::gxf::benchmark::BenchmarkState
DefaultError<nvidia::gxf::benchmark::BenchmarkState>() {
  return nvidia::gxf::benchmark::kBenchmarkStateUndefined;
}

}  // nvidia::expected_macro
