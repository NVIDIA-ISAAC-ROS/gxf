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

#include "gxf/serialization/file.hpp"
#include "gxf/std/codelet.hpp"

namespace nvidia {
namespace gxf {
namespace benchmark {
namespace test {

// Checks the validity of the specified benchmark report
class BenchmarkReportChecker : public gxf::Codelet {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t initialize() override { return GXF_SUCCESS; };
  gxf_result_t deinitialize() override;

  gxf_result_t start() override { return GXF_SUCCESS; }
  gxf_result_t tick() override { return GXF_SUCCESS; }
  gxf_result_t stop() override { return GXF_SUCCESS; }

 private:
  gxf::Parameter<gxf::Handle<gxf::File>> report_;
  gxf::Parameter<std::string> report_namespace_;
};

}  // namespace test
}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
