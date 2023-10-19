/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gxf/test/components/entity_monitor.hpp"

namespace nvidia {
namespace gxf {
namespace test {

gxf_result_t EntityMonitor::on_execute_abi(gxf_uid_t eid, uint64_t timestamp, gxf_result_t code) {
  // Get handle to entity
  auto entity = Entity::Shared(context(), eid);
  if (!entity) {
    return ToResultCode(entity);
  }

  // Convert timestamp from nanoseconds to milliseconds
  const double time_ms = static_cast<double>(timestamp) * 1e-6;

  // Log entity status to console
  GXF_LOG_INFO("[t = %0.1fms] %s: %s", time_ms, entity->name(), GxfResultStr(code));

  return GXF_SUCCESS;
}

}  // namespace test
}  // namespace gxf
}  // namespace nvidia
