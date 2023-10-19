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
#ifndef NVIDIA_GXF_CORE_SUBGRAPH_HPP
#define NVIDIA_GXF_CORE_SUBGRAPH_HPP

#include <string>

#include "gxf/core/gxf.h"

#include "common/assert.hpp"
#include "common/type_name.hpp"
#include "gxf/core/component.hpp"

namespace nvidia {
namespace gxf {


class Subgraph : public Component {
 public:
  gxf_result_t registerInterface(Registrar* registrar) {
    Expected<void> result;
    result &= registrar->parameter(location_, "location", "Yaml source of the subgraph", "");
    return ToResultCode(result);
  }

 private:
  Parameter<FilePath> location_;
};
}  // namespace  gxf
}  // namespace nvidia

#endif
