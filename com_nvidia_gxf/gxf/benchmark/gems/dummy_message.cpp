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
#include "gxf/benchmark/gems/dummy_message.hpp"

#include <string>
#include <utility>

namespace nvidia {
namespace gxf {
namespace benchmark {

namespace {
constexpr char const* kNameDummyComponentCount = "dummy_component_size";
constexpr char const* kNameTimestamp = "timestamp";
}

// Adds CameraMessage components to an entity
gxf::Expected<DummyMessageParts> CreateDummyMessage(
    gxf_context_t context, const uint64_t dummy_component_size) {
  DummyMessageParts message;

  return gxf::Entity::New(context)
      .assign_to(message.entity)
      .and_then([&]() {
        return message.entity.add<uint64_t>(kNameDummyComponentCount);
      })
      .map([&](gxf::Handle<uint64_t> dummy_component_size_handle) {
        *dummy_component_size_handle = dummy_component_size;
        message.dummy_component_size = dummy_component_size_handle;
        return gxf::Success;
      })
      .log_error("Failed to add '%s' to message entity", kNameDummyComponentCount)
      .and_then([&]() -> gxf::Expected<void> {
        for (uint64_t i = 0; i < dummy_component_size; i++) {
          auto new_component = message.entity.add<uint64_t>(std::string(
            "dummy_component_" + std::to_string(i)).c_str());
          if (!new_component) {
            GXF_LOG_ERROR("Failed to add dummy component %ld to message entity", i);
            return gxf::Unexpected{new_component.error()};
          }
          *new_component.value() = i;
          message.dummy_components.push_back(new_component.value());
        }
        return gxf::Success;
      })
      .and_then([&]() {
        return message.entity.add<gxf::Timestamp>(kNameTimestamp);
      })
      .assign_to(message.timestamp)
      .log_error("Failed to add '%s' to message entity", kNameTimestamp)
      .substitute(message);
}

gxf::Expected<DummyMessageParts> GetDummyMessage(gxf::Entity message) {
  DummyMessageParts parts;
  parts.entity = message;
  return parts.entity.get<uint64_t>(kNameDummyComponentCount)
      .assign_to(parts.dummy_component_size)
      .log_error("Failed to get '%s' from message entity", kNameDummyComponentCount)
      .and_then([&]() -> gxf::Expected<void> {
        for (uint64_t i = 0; i < *parts.dummy_component_size; i++) {
          auto dummy_component = parts.entity.get<uint64_t>(
            std::string("dummy_component_" + std::to_string(i)).c_str());
          if (!dummy_component) {
            GXF_LOG_ERROR("Failed to get dummy component %ld from message entity", i);
            return gxf::Unexpected{dummy_component.error()};
          }
          parts.dummy_components.push_back(dummy_component.value());
        }
        return gxf::Success;
      })
      .and_then([&]() {
        return parts.entity.get<gxf::Timestamp>(kNameTimestamp);
      })
      .assign_to(parts.timestamp)
      .log_error("Failed to get '%s' to message entity", kNameTimestamp)
      .substitute(parts);
}

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
