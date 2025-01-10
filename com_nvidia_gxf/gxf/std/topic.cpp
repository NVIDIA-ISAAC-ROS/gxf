/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gxf/std/topic.hpp"

#include "gxf/core/expected_macro.hpp"
#include "gxf/core/parameter_parser.hpp"
#include "gxf/core/parameter_parser_std.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t Topic::registerInterface(Registrar* registrar) {
  RETURN_IF_ERROR(registrar->parameter(topic_name_, "topic_name", "Topic Name"), {});
  RETURN_IF_ERROR(registrar->parameter(transmitters_, "transmitters", "Transmitters",
                                       "Transmitters that will be added to this topic.", {}));
  RETURN_IF_ERROR(registrar->parameter(receivers_, "receivers", "Receivers",
                                       "Receivers that will be added to this topic.", {}));
  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
