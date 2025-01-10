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
#include "gxf/benchmark/tests/benchmark_report_checker.hpp"

#include <unistd.h>
#include <string>

#include "gxf/benchmark/basic_metrics_calculator.hpp"
#include "nlohmann/json.hpp"

namespace nvidia {
namespace gxf {
namespace benchmark {
namespace test {

gxf_result_t BenchmarkReportChecker::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(
      report_, "report",
      "Benchmark report",
      "Benchmark report to check");
  result &= registrar->parameter(
      report_namespace_, "report_namespace",
      "Report's namespace",
      "Namespace of the report to be checked", std::string(""));
  return gxf::ToResultCode(result);
}

gxf_result_t BenchmarkReportChecker::deinitialize() {
  std::ifstream report_ifstream(report_->path());
  nlohmann::json loaded_report;
  try {
    loaded_report = nlohmann::json::parse(report_ifstream, nullptr, true);
  } catch (const nlohmann::json::parse_error& e) {
    GXF_LOG_ERROR("Failed to parse json file %s with error %s", report_->path(), e.what());
    return GXF_ARGUMENT_INVALID;
  }
  // Check if the report contains the vital BasicMetrics::kMeanOutputFrameRate metric
  if (!loaded_report.contains(
      GetNamespacedReportKey(
          BasicMetricsStrMap.at(BasicMetrics::kMeanOutputFrameRate),
          report_namespace_.get()))) {
    const std::string dump = loaded_report.dump(2);
    GXF_LOG_WARNING(
        "Mean output frame rate was not found from the exported report: \"%s\"",
        report_->path());
    GXF_LOG_WARNING(
        "The exported report:\r\n%s", dump.c_str());
    return GXF_SUCCESS;
  }

  // Check if the value of the mean output fps is reasonable (greater than 0.0fps)
  double mean_output_fps = loaded_report[
      GetNamespacedReportKey(
          BasicMetricsStrMap.at(BasicMetrics::kMeanOutputFrameRate),
          report_namespace_.get())];
  if (mean_output_fps <= 0.0) {
    GXF_LOG_ERROR("Mean output frame rate was not greater than 0: %f", mean_output_fps);
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

}  // namespace test
}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
