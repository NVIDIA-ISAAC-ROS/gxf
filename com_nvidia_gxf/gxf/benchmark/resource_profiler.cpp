/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "gxf/benchmark/resource_profiler.hpp"

namespace nvidia {
namespace gxf {
namespace benchmark {

gxf::Expected<void> ResourceProfiler::initializeHook() {
  GXF_LOG_INFO("Initializing resource profiler");
  RETURN_IF_ERROR(readProcStat(previous_cpu_jiffies_), "Failed to read CPU statistics");

  return gxf::Success;
}

gxf::Expected<void> ResourceProfiler::deinitializeHook() {
  return gxf::Success;
}

gxf::Expected<void> ResourceProfiler::readProcStat(std::vector<std::vector<int64_t>>& cpu_jiffies) {
  std::ifstream proc_stat;
  proc_stat.open("/proc/stat");
  if (!proc_stat.good()) {
    GXF_LOG_ERROR("Unable to open \"/proc/stat\" for reading");
    return Unexpected{GXF_FAILURE};
  }

  std::string line;
  while (std::getline(proc_stat, line)) {
    std::istringstream string_stream(line);
    if (string_stream.good()) {
      std::string start_word;
      // Eliminate "cpu" from the read strings of line
      string_stream >> start_word;
      size_t status = start_word.find("intr");
      // Parse only cpu usage related information
      if (status != std::string::npos) {
        GXF_LOG_DEBUG("Finished parsing for CPU usage information");
        break;
      }
    }

    std::vector<int64_t> read_cpu_jiffies((std::istream_iterator<int64_t>(string_stream)),
                                           std::istream_iterator<int64_t>());
    cpu_jiffies.push_back(read_cpu_jiffies);
  }

  proc_stat.close();
  return gxf::Success;
}

gxf::Expected<void> ResourceProfiler::beginHook() {
  // Get baseline resource usage
  RETURN_IF_ERROR(getCPUUtilization(baseline_overall_cpu_util_), "Failed to get CPU utilization");

  // Add a delay here so the next sampling is valid with a consistent time gap
  std::chrono::duration<double> duration_s(1.0/profile_sampling_rate_hz_.get());
  std::this_thread::sleep_for(duration_s);

  return gxf::Success;
}

gxf::Expected<void> ResourceProfiler::endHook() {
  return gxf::Success;
}

gxf::Expected<void> ResourceProfiler::getCPUUtilization(double& cpu_percentage) {
  std::vector<std::vector<int64_t>> current_cpu_jiffies;
  RETURN_IF_ERROR(readProcStat(current_cpu_jiffies), "Failed to read CPU statistics");

  std::vector<int64_t> previous_avg_jiffies = previous_cpu_jiffies_.front();
  std::vector<int64_t> current_avg_jiffies = current_cpu_jiffies.front();

  int64_t previous_sum_of_all_jiffies = 0;
  int64_t current_sum_of_all_jiffies = 0;
  for (size_t i = 0; i < current_avg_jiffies.size(); ++i) {
      previous_sum_of_all_jiffies += previous_avg_jiffies[i];
      current_sum_of_all_jiffies += current_avg_jiffies[i];
  }
  int64_t idle_jiffies = current_avg_jiffies[3] - previous_avg_jiffies[3];
  int64_t sum_jiffies = current_sum_of_all_jiffies - previous_sum_of_all_jiffies;
  cpu_percentage = 100.0 * (1.0 - (idle_jiffies / static_cast<double>(sum_jiffies)));
  std::swap(previous_cpu_jiffies_, current_cpu_jiffies);

  return gxf::Success;
}

gxf::Expected<void> ResourceProfiler::profileThreadHook() {
  double cpu_percentage = -1.0;
  RETURN_IF_ERROR(getCPUUtilization(cpu_percentage), "Failed to get CPU utilization");

  overall_cpu_util_samples_.push_back(cpu_percentage);

  return gxf::Success;
}

gxf::Expected<void> ResourceProfiler::resetHook() {
  return gxf::Success;
}

gxf::Expected<nlohmann::json> ResourceProfiler::computeHook() {
  nlohmann::json perf_results;

  // Overall CPU utilizations
  perf_results[kResourceMetricsStrMap.at(ResourceMetrics::kBaselineOverallCPUUtilization)] =
      baseline_overall_cpu_util_;
  const auto cpu_util_results = UNWRAP_OR_RETURN(
      ComputeMaxMinMeanSdValues<double>(overall_cpu_util_samples_));
  perf_results[kResourceMetricsStrMap.at(
      ResourceMetrics::kMaxOverallCPUUtilization)] = cpu_util_results.max;
  perf_results[kResourceMetricsStrMap.at(
      ResourceMetrics::kMinOverallCPUUtilization)] = cpu_util_results.min;
  perf_results[kResourceMetricsStrMap.at(
      ResourceMetrics::kMeanOverallCPUUtilization)] = cpu_util_results.mean;
  perf_results[kResourceMetricsStrMap.at(
      ResourceMetrics::kStdDevOverallCPUUtilization)] = cpu_util_results.sd;

  return perf_results;
}

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
