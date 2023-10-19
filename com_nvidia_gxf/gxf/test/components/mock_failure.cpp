/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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
#include "gxf/test/components/mock_failure.hpp"

namespace nvidia {
namespace gxf {
namespace test {

gxf_result_t MockFailure::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(num_max_ticks_, "num_max_ticks", "The number max ticks allowed");
  return ToResultCode(result);
}

gxf_result_t MockFailure::tick() {
  if (current_ticks_ >= num_max_ticks_) { return GXF_FAILURE; }
  current_ticks_++;
  return GXF_SUCCESS;
}

}  // namespace test
}  // namespace gxf
}  // namespace nvidia
