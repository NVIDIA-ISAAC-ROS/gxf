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

#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/scheduling_terms.hpp"


namespace nvidia {
namespace gxf {
namespace benchmark {
namespace test {

// Triggers AsynchronousSchedulingTerm based on data replay commands
class AsyncTriggerDataReplayer : public gxf::Codelet {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t start() override;
  gxf_result_t tick() override;

 private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> receiver_;
  gxf::Parameter<gxf::Handle<gxf::AsynchronousSchedulingTerm>>
      replay_data_source_async_scheduling_term_;
};

}  // namespace test
}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
