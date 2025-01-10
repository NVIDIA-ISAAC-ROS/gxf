/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_BENCHMARK_RESOURCE_PROFILER_HPP
#define NVIDIA_GXF_BENCHMARK_RESOURCE_PROFILER_HPP

#include <map>
#include <string>
#include <thread>
#include <vector>

#include "gxf/benchmark/resource_profiler_base.hpp"
#include "gxf/core/expected_macro.hpp"


namespace nvidia {
namespace gxf {
namespace benchmark {

// A profiler for measuring resource usage
class ResourceProfiler : public ResourceProfilerBase {
 public:
  // Initialize resource profiler
  gxf::Expected<void> initializeHook() override;

  // Deinitialize resource profiler
  gxf::Expected<void> deinitializeHook() override;

  // Start resource profiling thread
  gxf::Expected<void> beginHook() override;

  // Stop resource profiling thread
  gxf::Expected<void> endHook() override;

  // Hook function called in resource profiling thread
  gxf::Expected<void> profileThreadHook() override;

  // Reset all stored profiling history
  gxf::Expected<void> resetHook() override;

  // Compute profiling results for a benchmark iteration
  gxf::Expected<nlohmann::json> computeHook() override;

 private:
  gxf::Expected<void> readProcStat(std::vector<std::vector<int64_t>>& cpu_jiffies);
  gxf::Expected<void> getCPUUtilization(double& cpu_percentage);

  // vector to hold the values of previous jiffies
  // First line is the aggregate information of all the cpu cores followed by individual
  // core's usage information
  std::vector<std::vector<int64_t>> previous_cpu_jiffies_;
};

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_BENCHMARK_RESOURCE_PROFILER_HPP
