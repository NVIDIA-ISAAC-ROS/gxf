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
#include "gxf/benchmark/benchmark_controller.hpp"

#include <signal.h>

#include <unistd.h>
#include <string>
#include <vector>

namespace nvidia {
namespace gxf {
namespace benchmark {

#define STR_HELP(X) \
  case X:           \
    return #X;

const char* ToBenchmarkStateStr(BenchmarkState benchmark_state) {
  switch (benchmark_state) {
    STR_HELP(kBenchmarkStateStopped)
    STR_HELP(kBenchmarkStateInit)
    STR_HELP(kBenchmarkStateWaitingDependencies)
    STR_HELP(kBenchmarkStateStartBuffering)
    STR_HELP(kBenchmarkStateBuffering)
    STR_HELP(kBenchmarkStateStopBuffering)
    STR_HELP(kBenchmarkStateSetUpTestCase)
    STR_HELP(kBenchmarkStateStartRunningTestIteration)
    STR_HELP(kBenchmarkStateRunningTestIteration)
    STR_HELP(kBenchmarkStateSummarizeTestIteration)
    STR_HELP(kBenchmarkStateConcludeTestCase)
    STR_HELP(kBenchmarkStateExportReport)
    STR_HELP(kBenchmarkStateEnding)
    STR_HELP(kBenchmarkStateTerminating)
    STR_HELP(kBenchmarkStateUndefined)
    default:
      return "Undefined";
  }
}

#undef STR_HELP

gxf_result_t BenchmarkController::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  // Scheduling terms for benchmark flow
  result &= registrar->parameter(
      data_source_boolean_scheduling_terms_,
      "data_source_boolean_scheduling_terms",
      "Data source boolean scheduling terms",
      "Boolean scheduling terms for perminately stopping data sources",
      {});
  result &= registrar->parameter(
      data_source_async_scheduling_terms_, "data_source_async_scheduling_terms",
      "Data source async scheduling terms",
      "Scheduling terms to control execution of data source",
      {});
  result &= registrar->parameter(
      benchmark_controller_boolean_scheduling_term_,
      "benchmark_controller_boolean_scheduling_term",
      "Benchmark controller boolean scheduling term",
      "A boolean scheduling term for perminately stopping this controller");
  result &= registrar->parameter(
      benchmark_controller_target_time_scheduling_term_,
      "benchmark_controller_target_time_scheduling_term",
      "Benchmark controller target time scheduling term",
      "A target time scheduling term for enforcing timeout during benchmarking");
  result &= registrar->parameter(
      graph_boolean_scheduling_terms_,
      "graph_boolean_scheduling_terms",
      "Graph's boolean scheduling terms",
      "Boolean scheduling terms that will be disabled when benchmark ends",
      {});

  // Dependent components
  result &= registrar->parameter(
      dependent_components_, "dependent_components",
      "Dependent components",
      "A list of components whose states must be ready before starting to benchmark",
      {});

  // Transmitter to control the connected data replayer
  result &= registrar->parameter(
      data_replay_control_transmitter_, "data_replay_control_transmitter",
      "Transmitter of data replay control",
      "Transmitter to send replay command to the connected data replayer",
      gxf::Registrar::NoDefaultParameter(),
      GXF_PARAMETER_FLAGS_OPTIONAL);

  // Benchmark publishers and sinks
  result &= registrar->parameter(
      benchmark_publishers_, "benchmark_publishers",
      "Benchmark publishers",
      "A list of benchmark publishers that buffer and publish benchmark messages",
      {});
  result &= registrar->parameter(
      benchmark_sinks_, "benchmark_sinks",
      "Benchmark sinks",
      "A list of benchmark sinks that record message arrival timestamps");

  // Resource profilers
  result &= registrar->parameter(
      resource_profilers_,
      "resource_profilers",
      "Associated resource profilers",
      "A list of associated resource profilers to generate resource profiling reports",
      {});

  // Benchmark configurations
  result &= registrar->parameter(
      trial_run_, "trial_run",
      "Trial run switch",
      "Enable a trial run when set to true", true);
  result &= registrar->parameter(
      kill_at_the_end_, "kill_at_the_end",
      "Kill the benchmark process at the end",
      "Kill this process when the benchmark ends", false);
  result &= registrar->parameter(
      title_, "title",
      "Benchmark title",
      "Benchmark title to generate benchmark reports",
      std::string("Undefined Benchmark Title"));
  result &= registrar->parameter(
      exported_report_, "exported_report",
      "Exported report",
      "File to store exported report");
  result &= registrar->parameter(
      entity_buffer_size_, "entity_buffer_size",
      "Entity buffer size",
      "The number of messages to be buffered in each entity buffer", (size_t)5);
  result &= registrar->parameter(
      benchmark_duration_ms_, "benchmark_duration_ms",
      "Benchmark duration",
      "The duration of each benchmark iteration in miliseconds", (uint64_t)5000);
  result &= registrar->parameter(
      benchmark_iterations_, "benchmark_iterations",
      "The number of benchmark iterations",
      "The number of benchmark iterations to be conducted for each benchmark test case",
      (size_t)5);
  result &= registrar->parameter(
      benchmark_buffering_timeout_s_, "benchmark_buffering_timeout_s",
      "Benchmark buffering timeout",
      "The max wait time in seconds before stopping the benchmark buffering stage",
      static_cast<uint64_t>(5));
  result &= registrar->parameter(
      post_trial_benchmark_iteration_delay_s_, "post_trial_benchmark_iteration_delay_s",
      "Post trial benchmark iteration delay",
      "The wait time in seconds after a trial benchmark iteration before summarizing results",
      static_cast<uint64_t>(2));
  result &= registrar->parameter(
      post_benchmark_iteration_delay_s_, "post_benchmark_iteration_delay_s",
      "Post benchmark iteration delay",
      "The wait time in seconds after each benchmark iteration before summarizing results",
      static_cast<uint64_t>(2));
  return gxf::ToResultCode(result);
}

gxf_result_t BenchmarkController::start() {
  for (auto benchmark_publisher : benchmark_publishers_.get()) {
    benchmark_publisher->getAsyncSchedulingterm()->setEventState(
        nvidia::gxf::AsynchronousEventState::WAIT);
  }
  for (auto data_source_async_scheduling_term : data_source_async_scheduling_terms_.get()) {
    data_source_async_scheduling_term->setEventState(
        nvidia::gxf::AsynchronousEventState::WAIT);
  }
  // Ensure that the controller can tick the first time
  benchmark_controller_target_time_scheduling_term_->setNextTargetTime(
      getExecutionTimestamp());
  benchmark_state_ = BenchmarkState::kBenchmarkStateInit;
  last_benchmark_state_ = BenchmarkState::kBenchmarkStateStopped;
  return GXF_SUCCESS;
}

gxf_result_t BenchmarkController::tick() {
  BenchmarkState next_benchmark_state = BenchmarkState::kBenchmarkStateUndefined;

  if (benchmark_state_ != last_benchmark_state_) {
    GXF_LOG_INFO("Entering the benchmark state: \"%s\"",
                 ToBenchmarkStateStr(benchmark_state_));
  }

  switch (benchmark_state_) {
    case BenchmarkState::kBenchmarkStateInit:
      // Set up initial states for benchmarking
      next_benchmark_state = UNWRAP_OR_RETURN(handleBenchmarkStateInit());
      break;
    case BenchmarkState::kBenchmarkStateWaitingDependencies:
      // Wait for dependent components to finish their start()
      next_benchmark_state =
          UNWRAP_OR_RETURN(handleBenchmarkStateWaitingDependencies());
      break;
    case BenchmarkState::kBenchmarkStateStartBuffering:
      // Signal to start buffering
      next_benchmark_state = UNWRAP_OR_RETURN(handleBenchmarkStateStartBuffering());
      break;
    case BenchmarkState::kBenchmarkStateBuffering:
      // Buffering benchmark messages
      next_benchmark_state = UNWRAP_OR_RETURN(handleBenchmarkStateBuffering());
      break;
    case BenchmarkState::kBenchmarkStateStopBuffering:
      // Signal to stop buffering
      next_benchmark_state = UNWRAP_OR_RETURN(handleBenchmarkStateStopBuffering());
      break;
    case BenchmarkState::kBenchmarkStateSetUpTestCase:
      // Preparing for a new benchmark test case
      next_benchmark_state = UNWRAP_OR_RETURN(handleBenchmarkStateSetUpTestCase());
      break;
    case BenchmarkState::kBenchmarkStateStartRunningTestIteration:
      // Starting a new benchmark iteration
      next_benchmark_state =
          UNWRAP_OR_RETURN(handleBenchmarkStateStartRunningTestIteration());
      break;
    case BenchmarkState::kBenchmarkStateRunningTestIteration:
      // Running a benchmark iteration
      next_benchmark_state =
          UNWRAP_OR_RETURN(handleBenchmarkStateRunningTestIteration());
      break;
    case BenchmarkState::kBenchmarkStateSummarizeTestIteration:
      // Summarizing the current benchmark iteration
      next_benchmark_state =
          UNWRAP_OR_RETURN(handleBenchmarkStateSummarizeTestIteration());
      break;
    case BenchmarkState::kBenchmarkStateConcludeTestCase:
      // Concluding the current benchmark test case
      next_benchmark_state =
          UNWRAP_OR_RETURN(handleBenchmarkStateConcludeTestCase());
      break;
    case BenchmarkState::kBenchmarkStateExportReport:
      // Exporting the final report
      next_benchmark_state = UNWRAP_OR_RETURN(handleBenchmarkStateExportReport());
      break;
    case BenchmarkState::kBenchmarkStateEnding:
     // Ending the benchmark controller
      next_benchmark_state = UNWRAP_OR_RETURN(handleBenchmarkStateEnding());
      break;
    case BenchmarkState::kBenchmarkStateTerminating:
      // Terminating the benchmark process
      handleBenchmarkStateTerminating();
      break;
    default:
      GXF_LOG_ERROR("Unknown benchmark state");
      return GXF_FAILURE;
  }

  // Set the next target scheduling time point.
  // If next_timeout_timestamp_ns_ is not updated (i.e., expired) by any state handler,
  // next_timeout_timestamp_ns_ will be refreshed to ensure that the controller can
  // tick immediately
  if (next_timeout_timestamp_ns_ < getExecutionTimestamp()) {
    next_timeout_timestamp_ns_ = getExecutionTimestamp();
  }
  benchmark_controller_target_time_scheduling_term_->setNextTargetTime(
      next_timeout_timestamp_ns_);

  last_benchmark_state_ = benchmark_state_;
  benchmark_state_ = next_benchmark_state;
  return GXF_SUCCESS;
}

gxf::Expected<BenchmarkState> BenchmarkController::handleBenchmarkStateInit() {
  GXF_LOG_INFO("Setting up benchmark initial states");
  if (data_replay_control_transmitter_.try_get()) {
    GXF_LOG_INFO("Pausing the connected data replayer");
    RETURN_IF_ERROR(pauseDataReplay());
  }
  if (benchmark_publishers_.get().size() > 0) {
    GXF_LOG_INFO(
        "Set to buffer %ld messages per entity buffer", entity_buffer_size_.get());
    for (auto benchmark_publisher : benchmark_publishers_.get()) {
      benchmark_publisher->getEntityBuffer()->setMaxBufferSize(
          entity_buffer_size_.get());
    }
  }
  if (dependent_components_.get().size() > 0) {
    GXF_LOG_INFO("Waiting for dependent components to be ready");
    return BenchmarkState::kBenchmarkStateWaitingDependencies;
  }
  if (benchmark_publishers_.get().size() == 0) {
    return BenchmarkState::kBenchmarkStateSetUpTestCase;
  }
  return BenchmarkState::kBenchmarkStateStartBuffering;
}

gxf::Expected<BenchmarkState>
BenchmarkController::handleBenchmarkStateWaitingDependencies() {
  if (dependent_components_.get().size() > 0) {
    gxf_entity_status_t entity_status;
    for (const auto & component_handle : dependent_components_.get()) {
      const gxf_result_t result = GxfEntityGetStatus(
          this->context(), component_handle->eid(), &entity_status);
      if (result != GXF_SUCCESS) {
        GXF_LOG_ERROR("Failed to check status of dependent components");
        return gxf::Unexpected{GXF_FAILURE};
      }
      if (entity_status == gxf_entity_status_t::GXF_ENTITY_STATUS_NOT_STARTED ||
          entity_status == gxf_entity_status_t::GXF_ENTITY_STATUS_START_PENDING) {
        return BenchmarkState::kBenchmarkStateWaitingDependencies;
      }
    }
  }
  GXF_LOG_INFO("All dependent components are ready");
  if (benchmark_publishers_.get().size() == 0) {
    return BenchmarkState::kBenchmarkStateSetUpTestCase;
  }
  return BenchmarkState::kBenchmarkStateStartBuffering;
}

gxf::Expected<BenchmarkState> BenchmarkController::handleBenchmarkStateStartBuffering() {
  if (data_replay_control_transmitter_.try_get()) {
    GXF_LOG_INFO("Signal the connected data replayer to play messages");
    RETURN_IF_ERROR(playDataReplay());
  }
  if (data_source_async_scheduling_terms_.get().size() > 0) {
    GXF_LOG_INFO("Enable data sources' async scheduling terms");
    for (auto data_source_async_scheduling_term : data_source_async_scheduling_terms_.get()) {
      data_source_async_scheduling_term->setEventState(
          nvidia::gxf::AsynchronousEventState::EVENT_DONE);
    }
  }
  next_timeout_timestamp_ns_ =
      getExecutionTimestamp() + benchmark_buffering_timeout_s_.get()*1000000000;
  return BenchmarkState::kBenchmarkStateBuffering;
}

gxf::Expected<BenchmarkState> BenchmarkController::handleBenchmarkStateBuffering() {
  if (areAllBenchmarkPublisherBuffersFull()) {
    GXF_LOG_INFO("All %ld messages were buffered", entity_buffer_size_.get());

    // Stop the data replayer if present
    if (data_replay_control_transmitter_.try_get()) {
      GXF_LOG_INFO("Pause the connected data replayer");
      RETURN_IF_ERROR(pauseDataReplay());
    }

    return BenchmarkState::kBenchmarkStateStopBuffering;
  }

  bool need_to_terminate = false;

  // Check if it's timed out
  if (getExecutionTimestamp() > next_timeout_timestamp_ns_) {
    // Timeout with failed buffering
    GXF_LOG_ERROR("Timed out for buffering messages");
    need_to_terminate = true;
  }

  // Check data sources' async scheduling terms are still active
  for (auto data_source_async_scheduling_term : data_source_async_scheduling_terms_.get()) {
    bool is_data_source_event_done =
        data_source_async_scheduling_term->getEventState() ==
        nvidia::gxf::AsynchronousEventState::EVENT_DONE;
    if (!is_data_source_event_done) {
      GXF_LOG_ERROR("Data source's async scheduling term (%s/%s) was disabled before "
                    "all messages were buffered",
                    data_source_async_scheduling_term->entity().name(),
                    data_source_async_scheduling_term.name());
      need_to_terminate = true;
    }
  }

  // Check if data sources are still active
  for (auto data_source_boolean_scheduling_term : data_source_boolean_scheduling_terms_.get()) {
    if (!data_source_boolean_scheduling_term->checkTickEnabled()) {
      GXF_LOG_ERROR("Data source's boolean scheduling term (%s/%s) was disabled before "
                    "all messages were buffered",
                    data_source_boolean_scheduling_term->entity().name(),
                    data_source_boolean_scheduling_term.name());
    }
    need_to_terminate = true;
  }

  // Print buffer's current states and terminate if failed during buffering
  if (need_to_terminate) {
    for (auto benchmark_publisher : benchmark_publishers_.get()) {
      GXF_LOG_ERROR("\t\"%s/%s\" buffered %ld/%ld messages",
          benchmark_publisher->entity().name(),
          benchmark_publisher.name(),
          benchmark_publisher->getEntityBuffer()->getBuffer().size(),
          entity_buffer_size_.get());
    }
    return BenchmarkState::kBenchmarkStateTerminating;
  }

  // Keep waiting for all messages to be buffered
  return BenchmarkState::kBenchmarkStateBuffering;
}

gxf::Expected<BenchmarkState> BenchmarkController::handleBenchmarkStateStopBuffering() {
  // Stop the data source and sleep to give it time to end
  if (data_source_async_scheduling_terms_.get().size() > 0) {
    GXF_LOG_INFO("Disable data sources' async scheduling terms");
    for (auto data_source_async_scheduling_term : data_source_async_scheduling_terms_.get()) {
      data_source_async_scheduling_term->setEventState(
          nvidia::gxf::AsynchronousEventState::WAIT);
    }
  }
  if (data_source_boolean_scheduling_terms_.get().size() > 0) {
    GXF_LOG_INFO("Disable data sources' boolean scheduling terms");
    for (auto data_source_boolean_scheduling_term : data_source_boolean_scheduling_terms_.get()) {
      data_source_boolean_scheduling_term->disable_tick();
    }
  }
  sleep(2);

  return BenchmarkState::kBenchmarkStateSetUpTestCase;
}

gxf::Expected<BenchmarkState> BenchmarkController::handleBenchmarkStateSetUpTestCase() {
  benchmark_iteration_count_ = 0;
  if (trial_run_.get() == true) {
    // Enable trial run
    is_trial_running_ = true;
  } else {
    is_trial_running_ = false;
  }
  auto reset_result = resetResourceProfilers();
  if (!reset_result) {
    return gxf::ForwardError(reset_result);
  }
  reset_result = resetBenchmarkSinks();
  if (!reset_result) {
    return gxf::ForwardError(reset_result);
  }
  return BenchmarkState::kBenchmarkStateStartRunningTestIteration;
}

gxf::Expected<BenchmarkState>
BenchmarkController::handleBenchmarkStateStartRunningTestIteration() {
  if (benchmark_iteration_count_ >= benchmark_iterations_.get()) {
    GXF_LOG_INFO(
        "All %ld benchmark iterations were finished", benchmark_iteration_count_);
    return BenchmarkState::kBenchmarkStateConcludeTestCase;
  }

  if (is_trial_running_) {
    // This next benchmark iteration will be a trial run
    GXF_LOG_INFO("Running trial benchmark");
  } else {
    if (benchmark_iteration_count_ == 0) {
      // Reset resource profilers and sinks before first time starting a valid iteration
      auto reset_result = resetResourceProfilers();
      if (!reset_result) {
        return gxf::ForwardError(reset_result);
      }
      reset_result = resetBenchmarkSinks();
      if (!reset_result) {
        return gxf::ForwardError(reset_result);
      }
    }
    benchmark_iteration_count_++;
    GXF_LOG_INFO("Running #%ld benchmark iteration", benchmark_iteration_count_);
  }

  clearBenchmarkPublishers();
  clearBenchmarkSinks();

  // Inform that a benchmark iteration is starting
  auto signal_result = signalBeginBenchmarking();
  if (!signal_result) {
    GXF_LOG_ERROR("Failed to signal to begin a benchmark iteration");
    return gxf::Unexpected{GXF_FAILURE};
  }

  if (benchmark_publishers_.get().size() > 0) {
    GXF_LOG_INFO(
        "Signal benchmark publisher(s) to send out benchmark messages (duration = %ldms)",
        benchmark_duration_ms_.get());
    for (auto benchmark_publisher : benchmark_publishers_.get()) {
      // Enable the benchmark publisher's ticks
      benchmark_publisher->getAsyncSchedulingterm()->setEventState(
          nvidia::gxf::AsynchronousEventState::EVENT_DONE);
    }
  }

  next_timeout_timestamp_ns_ =
      getExecutionTimestamp() + benchmark_duration_ms_.get()*1000000;
  return BenchmarkState::kBenchmarkStateRunningTestIteration;
}

gxf::Expected<BenchmarkState>
BenchmarkController::handleBenchmarkStateRunningTestIteration() {
  if (getExecutionTimestamp() < next_timeout_timestamp_ns_) {
    // Keep benchmarking until benchmark duration is passed
    return BenchmarkState::kBenchmarkStateRunningTestIteration;
  }

  // Inform resource profilers that a benchmark iteration is ending
  auto signal_result = signalEndBenchmarkingToResourceProfilers();
  if (!signal_result) {
    GXF_LOG_ERROR("Failed to signal to end a benchmark iteration");
    return gxf::Unexpected{GXF_FAILURE};
  }

  // Stop all the benchmark publishers and sleep to give it time to end
  if (benchmark_publishers_.get().size() > 0) {
    GXF_LOG_INFO("Stop all benchmark publishers");
    stopBenchmarkPublishers();
  }
  if (is_trial_running_) {
    sleep(post_trial_benchmark_iteration_delay_s_.get());
  } else {
    sleep(post_benchmark_iteration_delay_s_.get());
  }

  // Inform sinks that a benchmark iteration is ending
  signal_result = signalEndBenchmarkingToBenchmarkSinks();
  if (!signal_result) {
    GXF_LOG_ERROR("Failed to signal to end a benchmark iteration");
    return gxf::Unexpected{GXF_FAILURE};
  }

  return BenchmarkState::kBenchmarkStateSummarizeTestIteration;
}

gxf::Expected<BenchmarkState>
BenchmarkController::handleBenchmarkStateSummarizeTestIteration() {
  auto maybe_report = summarizeCurrentTestIteration();
  if (!maybe_report) {
    GXF_LOG_ERROR("Failed to summarize a benchmark iteration");
    return gxf::ForwardError(maybe_report);
  }
  nlohmann::json iteration_perf_report = maybe_report.value()[0];
  std::string sub_heading;
  if (is_trial_running_) {
    sub_heading = "Trial Run";
    is_trial_running_ = false;
  } else {
    sub_heading = "#" + std::to_string(benchmark_iteration_count_);
  }
  printReport(iteration_perf_report, sub_heading);
  return BenchmarkState::kBenchmarkStateStartRunningTestIteration;
}

gxf::Expected<BenchmarkState>
BenchmarkController::handleBenchmarkStateConcludeTestCase() {
  auto maybe_report = concludeTestCase();
  if (!maybe_report) {
    GXF_LOG_ERROR("Failed to conclude a benchmark test case");
    return gxf::ForwardError(maybe_report);
  }
  report_ = maybe_report.value()[0];
  printReport(report_, "Final Report");
  return BenchmarkState::kBenchmarkStateExportReport;
}

gxf::Expected<BenchmarkState> BenchmarkController::handleBenchmarkStateExportReport() {
  GXF_LOG_INFO("Exporting the final report to \"%s\"", exported_report_->path());
  const std::string dump = report_.dump(2);
  exported_report_->write(dump.data(), dump.size());
  exported_report_->close();
  return BenchmarkState::kBenchmarkStateEnding;
}

gxf::Expected<BenchmarkState> BenchmarkController::handleBenchmarkStateEnding() {
  if (graph_boolean_scheduling_terms_.get().size() > 0) {
    GXF_LOG_INFO("Stopping graph boolean scheduling terms");
    for (auto boolean_scheduling_term : graph_boolean_scheduling_terms_.get()) {
      boolean_scheduling_term->disable_tick();
    }
  }
  GXF_LOG_INFO("Stopping benchmark controller");
  benchmark_controller_boolean_scheduling_term_->disable_tick();
  if (kill_at_the_end_.get()) {
    GXF_LOG_INFO("Killing the current benchmark process (pid=%d)", ::getpid());
    ::kill(::getpid(), SIGINT);
  }
  return BenchmarkState::kBenchmarkStateEnding;
}

void BenchmarkController::handleBenchmarkStateTerminating() {
  GXF_LOG_ERROR("Forcing to terminate the benchmark process");
  ::exit(EXIT_FAILURE);
}

gxf::Expected<nlohmann::json> BenchmarkController::summarizeCurrentTestIteration() {
  nlohmann::json perf_report;
  // Performance calculators
  for (auto benchmark_sink : benchmark_sinks_.get()) {
    nlohmann::json this_perf_report =
        UNWRAP_OR_RETURN(benchmark_sink->compute())[0];
    if (this_perf_report.is_object()) {
      perf_report.merge_patch(this_perf_report);
    }
  }
  // Resource profilers
  for (auto &resource_profiler : resource_profilers_.get()) {
    nlohmann::json this_perf_report =
        UNWRAP_OR_RETURN(resource_profiler->compute())[0];
    if (this_perf_report.is_object()) {
      perf_report.merge_patch(this_perf_report);
    }
  }
  return perf_report;
}

bool BenchmarkController::areAllBenchmarkPublisherBuffersFull() const {
  for (auto benchmark_publisher : benchmark_publishers_.get()) {
    if (!benchmark_publisher->getEntityBuffer()->isBufferFull()) {
      return false;
    }
  }
  return true;
}

gxf::Expected<nlohmann::json> BenchmarkController::concludeTestCase() {
  nlohmann::json test_case_perf_report;
  // Resource profilers
  for (auto &resource_profiler : resource_profilers_.get()) {
    nlohmann::json this_perf_report = resource_profiler->conclude();
    if (this_perf_report.is_object()) {
      test_case_perf_report.merge_patch(this_perf_report);
    }
  }
  // Benchmark sinks
  for (auto benchmark_sink : benchmark_sinks_.get()) {
    nlohmann::json this_perf_report = benchmark_sink->conclude();
    if (this_perf_report.is_object()) {
      test_case_perf_report.merge_patch(this_perf_report);
    }
  }
  return test_case_perf_report;
}

gxf::Expected<void> BenchmarkController::signalBeginBenchmarking() {
  // Benchmark sinks
  for (auto benchmark_sink : benchmark_sinks_.get()) {
    RETURN_IF_ERROR(
        benchmark_sink->begin(),
        "Failed to signal a benchmark sink to begin benchmarking");
  }
  // Resource profilers
  for (auto &resource_profiler : resource_profilers_.get()) {
    RETURN_IF_ERROR(
        resource_profiler->begin(),
        "Failed to signal a resource profiler to begin benchmarking");
  }
  return gxf::Success;
}

gxf::Expected<void> BenchmarkController::signalEndBenchmarkingToResourceProfilers() {
  for (auto &resource_profiler : resource_profilers_.get()) {
    RETURN_IF_ERROR(
        resource_profiler->end(),
        "Failed to signal a resource profiler to end benchmarking");
  }
  return gxf::Success;
}

gxf::Expected<void> BenchmarkController::signalEndBenchmarkingToBenchmarkSinks() {
  for (auto benchmark_sink : benchmark_sinks_.get()) {
    RETURN_IF_ERROR(
        benchmark_sink->end(),
        "Failed to signal a benchmark sink to end benchmarking");
  }
  return gxf::Success;
}

gxf::Expected<void> BenchmarkController::resetResourceProfilers() {
  for (auto &resource_profiler : resource_profilers_.get()) {
    RETURN_IF_ERROR(
        resource_profiler->reset(),
        "Failed to reset a resource profiler");
  }
  return gxf::Success;
}

gxf::Expected<void> BenchmarkController::resetBenchmarkSinks() {
  for (auto benchmark_sink : benchmark_sinks_.get()) {
    RETURN_IF_ERROR(
        benchmark_sink->reset(),
        "Failed to reset a benchmark sink");
  }
  return gxf::Success;
}

void BenchmarkController::clearBenchmarkSinks() {
  for (auto benchmark_sink : benchmark_sinks_.get()) {
    benchmark_sink->clearRecordedTimestamps();
  }
}

void BenchmarkController::clearBenchmarkPublishers() {
  for (auto benchmark_publisher : benchmark_publishers_.get()) {
    benchmark_publisher->clearRecordedTimestamps();
  }
}

void BenchmarkController::stopBenchmarkPublishers() {
  for (auto benchmark_publisher : benchmark_publishers_.get()) {
    benchmark_publisher->getAsyncSchedulingterm()->setEventState(
      nvidia::gxf::AsynchronousEventState::WAIT);
  }
}

std::string GetCenteredString(const std::string& str, const size_t width) {
  std::stringstream output_str;
  size_t output_str_center_setw_value = ceil(width/2.0 + str.size()/2.0);
  output_str << std::setw(output_str_center_setw_value) << str;
  for (size_t i = 0; i < (width - output_str_center_setw_value); i++) {
    output_str << " ";
  }
  return output_str.str();
}

void PrintTableLine(const size_t line_width) {
  std::stringstream output_str;
  output_str << std::setfill('-') << std::setw(line_width+2) << "-";
  std::printf("+%s+\n", output_str.str().c_str());
}

void PrintHeadingLines(
    const std::string& heading,
    const std::string& sub_heading,
    const size_t line_width) {
  std::printf("| %s |\n", GetCenteredString(heading, line_width).c_str());
  if (!sub_heading.empty()) {
    std::printf("| %s |\n", GetCenteredString(sub_heading, line_width).c_str());
  }
}

void BenchmarkController::printReport(
    nlohmann::json report, std::string sub_heading = "") {
  std::vector<std::string> str_buffer;
  size_t max_line_width = 0;
  std::string heading = title_.get();

  // Buffer lines before printing to figure out the maximum line width needed
  for (const auto& item : report.items()) {
    std::string current_line = item.key() + ": ";
    if (item.value().is_number_float()) {
      std::stringstream str_stream;
      str_stream << std::fixed << std::setprecision(3) << static_cast<double>(item.value());
      current_line += str_stream.str();
    } else {
      current_line += nlohmann::to_string(item.value());
    }
    max_line_width = current_line.size() > max_line_width ? current_line.size() : max_line_width;
    str_buffer.push_back(current_line);
  }
  max_line_width = heading.size() > max_line_width ? heading.size() : max_line_width;
  max_line_width = sub_heading.size() > max_line_width ? sub_heading.size() : max_line_width;

  // Start printing the report
  PrintTableLine(max_line_width);
  PrintHeadingLines(heading, sub_heading, max_line_width);
  PrintTableLine(max_line_width);
  for (const std::string& line : str_buffer) {
    std::stringstream output_line;
    output_line << "| " << std::left << std::setw(max_line_width) << line << " |";
    std::printf("%s\n", output_line.str().c_str());
  }
  PrintTableLine(max_line_width);
}

gxf::Expected<void> BenchmarkController::sendDataReplayCommand(
    nvidia::gxf::benchmark::DataReplayControl::Command command) {
  if (data_replay_control_transmitter_.try_get()) {
    gxf::Entity data_replay_control_msg = UNWRAP_OR_RETURN(gxf::Entity::New(context()));
    auto data_replay_control = UNWRAP_OR_RETURN(data_replay_control_msg.add<DataReplayControl>());
    data_replay_control->replay_command = command;
    RETURN_IF_ERROR(data_replay_control_transmitter_.try_get().value()->publish(
        data_replay_control_msg, getExecutionTimestamp()));
  } else {
    GXF_LOG_ERROR("No data replayer was connected");
    return gxf::Unexpected{GXF_FAILURE};
  }
  return gxf::Success;
}

gxf::Expected<void> BenchmarkController::pauseDataReplay() {
  return sendDataReplayCommand(
      nvidia::gxf::benchmark::DataReplayControl::Command::kPause);
}

gxf::Expected<void> BenchmarkController::playDataReplay() {
  return sendDataReplayCommand(
      nvidia::gxf::benchmark::DataReplayControl::Command::kPlay);
}

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
