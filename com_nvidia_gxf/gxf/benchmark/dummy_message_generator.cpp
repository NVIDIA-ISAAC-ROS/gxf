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

#include "gxf/benchmark/dummy_message_generator.hpp"

#include <chrono>
#include <utility>

#include "common/nvtx_helper.hpp"
#include "gxf/benchmark/gems/dummy_message.hpp"

namespace nvidia {
namespace gxf {
namespace benchmark {

nvtxEventAttributes_t CreateYellowEvent(const char * message, uint32_t category) {
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = 0xFFFFFF00;  // Yellow
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = message;
  eventAttrib.category = category;

  return eventAttrib;
}

gxf_result_t DummyMessageGenerator::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(
      dummy_component_size_, "dummy_component_size", "Dummy Component Count",
      "Number of dummy components to generate.", (uint64_t)5);
  result &= registrar->parameter(
      publish_new_message_, "publish_new_message", "Publish New Dummy Messages",
      "If true, create new dummy messages to publish otherwise forward messages from receiver.",
      true);
  result &= registrar->parameter(
      receiver_, "receiver", "Receiver",
      "Handle to the receiver for receiving messages.",
      gxf::Registrar::NoDefaultParameter(),
      GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      transmitter_, "transmitter", "Transmitter",
      "Handle to the transmitter for sending messages.");
  return gxf::ToResultCode(result);
}

gxf_result_t DummyMessageGenerator::tick() {
  DummyMessageParts received_dummy_message_parts;
  DummyMessageParts out_dummy_message_parts;
  bool is_received_message = false;

  if (receiver_.try_get()) {
    // Get a message from the receiver if connected
    {
      auto nvtx_attribute = CreateYellowEvent(
        "DummyMessageGenerator: receiver_.try_get()->get()->receive()", 0);
      nvtxRangePushEx(&nvtx_attribute);
    }
    auto received_message = receiver_.try_get()->get()->receive();
    nvtxRangePop();
    if (received_message) {
      // If successful, create a new message entity with the same number of dummy components
      {
        auto nvtx_attribute = CreateYellowEvent(
          "DummyMessageGenerator: GetDummyMessage(received_message.value())", 0);
        nvtxRangePushEx(&nvtx_attribute);
      }
      auto maybe_received_dummy_message_parts = GetDummyMessage(received_message.value());
      nvtxRangePop();
      if (!maybe_received_dummy_message_parts) {
        GXF_LOG_ERROR("Failed to get dummy message parts from received message");
        return maybe_received_dummy_message_parts.error();
      }
      received_dummy_message_parts = maybe_received_dummy_message_parts.value();
      is_received_message = true;
    }
  }

  // Create a new dummy message entity to be published
  if (publish_new_message_.get()) {
    auto final_dummy_component_size = is_received_message ?
        *received_dummy_message_parts.dummy_component_size.get() : dummy_component_size_;
    {
      auto nvtx_attribute = CreateYellowEvent(
        "DummyMessageGenerator: CreateDummyMessage(context(), final_dummy_component_size)", 0);
      nvtxRangePushEx(&nvtx_attribute);
    }
    auto maybe_out_dummy_message_parts = CreateDummyMessage(context(), final_dummy_component_size);
    nvtxRangePop();
    if (!maybe_out_dummy_message_parts) {
      GXF_LOG_ERROR("Failed to create a new dummy message parts");
      return maybe_out_dummy_message_parts.error();
    }
    out_dummy_message_parts = maybe_out_dummy_message_parts.value();

    if (is_received_message) {
      // Copy dummy components from the received dummy message to output dummy message
      {
        auto nvtx_attribute = CreateYellowEvent(
          "DummyMessageGenerator: Copy component values from received to new", 0);
        nvtxRangePushEx(&nvtx_attribute);
      }
      for (size_t i = 0; i < received_dummy_message_parts.dummy_components.size(); i++) {
        // Intentionally copy the dummy components one by one to mimic the behavior of an
        // actual codelet filling up all the component values.
        out_dummy_message_parts.dummy_components[i] =
            received_dummy_message_parts.dummy_components[i];
      }
      nvtxRangePop();
    }

    // Publish the newly created message
    {
      auto nvtx_attribute = CreateYellowEvent(
        "DummyMessageGenerator: transmitter_->publish(out_dummy_message_parts.entity)", 0);
      nvtxRangePushEx(&nvtx_attribute);
    }
    auto result = transmitter_->publish(out_dummy_message_parts.entity);
    nvtxRangePop();
    return gxf::ToResultCode(result);
  }

  // Forward the received message
  {
    auto nvtx_attribute = CreateYellowEvent(
      "DummyMessageGenerator: transmitter_->publish(received_dummy_message_parts.entity)", 0);
    nvtxRangePushEx(&nvtx_attribute);
  }
  auto result = transmitter_->publish(received_dummy_message_parts.entity);
  nvtxRangePop();
  return gxf::ToResultCode(result);
}

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
