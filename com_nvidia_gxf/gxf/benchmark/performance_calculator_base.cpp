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
#include "gxf/benchmark/performance_calculator_base.hpp"

#include <string>

namespace nvidia {
namespace gxf {
namespace benchmark {

std::string GetNamespacedReportKey(
    const std::string& key,
    const std::string& prefix) {
  if (prefix.empty()) {
    return key;
  }
  return "[" + prefix + "] " + key;
}

nlohmann::json GetNamespacedReport(
    const nlohmann::json& report,
    const std::string& prefix) {
  if (prefix.empty()) {
    return report;
  }
  nlohmann::json namespaced_report;
  for (const auto& item : report.items()) {
    namespaced_report[GetNamespacedReportKey(item.key(), prefix)] = item.value();
  }
  return namespaced_report;
}

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
