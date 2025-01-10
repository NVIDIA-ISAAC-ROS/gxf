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

#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {
namespace benchmark {

// Generates dummy messages.
class DummyMessageGenerator : public gxf::Codelet {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t start() override { return GXF_SUCCESS; }
  gxf_result_t tick() override;
  gxf_result_t stop() override { return GXF_SUCCESS; }

 private:
  gxf::Parameter<uint64_t> dummy_component_size_;
  gxf::Parameter<bool> publish_new_message_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> receiver_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> transmitter_;
};

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
