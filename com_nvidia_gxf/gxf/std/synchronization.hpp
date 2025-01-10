/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_GXF_CORE_SYNCHRONIZATION_HPP
#define NVIDIA_GXF_CORE_SYNCHRONIZATION_HPP

#include <string>
#include <utility>
#include <vector>

#include "gxf/core/gxf.h"

#include "common/assert.hpp"
#include "common/type_name.hpp"
#include "gxf/core/component.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

// A codelet to synchronize incoming messages from multiple receivers and
// publish them on to multiple transmitters. The number of receivers and transmitters
// must be equal in number and incoming messages must have a valid nvidia::gxf::Timestamp
// component in them. Only one message per transmitter is published per tick of the codelet.

class Synchronization : public Codelet {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t start() override;
  gxf_result_t tick() override;

 private:
  Parameter<std::vector<Handle<Receiver>>> inputs_;
  Parameter<std::vector<Handle<Transmitter>>> outputs_;
  Parameter<int64_t> sync_threshold_;
};

}  // namespace  gxf
}  // namespace nvidia

#endif
