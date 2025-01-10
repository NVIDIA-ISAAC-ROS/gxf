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

#include <vector>

#include "gxf/core/entity.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace gxf {
namespace benchmark {

struct DummyMessageParts {
  // The message entity
  gxf::Entity entity;
  // Dummy component count
  gxf::Handle<uint64_t> dummy_component_size;
  // Dummy components
  std::vector<gxf::Handle<uint64_t>> dummy_components;
  // Timestamp of publishing and acquisition in system time
  gxf::Handle<gxf::Timestamp> timestamp;
};

gxf::Expected<DummyMessageParts> CreateDummyMessage(
    gxf_context_t context, const uint64_t dummy_component_size);

gxf::Expected<DummyMessageParts> GetDummyMessage(gxf::Entity message);

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
