/*
Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <vector>

#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/std/gather.hpp"
namespace nvidia {
namespace gxf {

gxf_result_t Gather::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &=
      registrar->parameter(sink_, "sink", "Sink", "The output channel for gathered messages.");
  result &=
      registrar->parameter(sources_, "sources", "Sources", "The input channels for gathering "
                           "messages.", Registrar::NoDefaultParameter(),
                           GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter<int64_t>(
      tick_source_limit_, "tick_source_limit", "Tick Source Limit",
      "Maximum number of messages to take from each source in one tick. 0 means no limit.", 0);
  return ToResultCode(result);
}

gxf_result_t Gather::tick() {
  // Get all messages from all receivers and put them onto the output channel.
  const auto receivers = entity().findAllHeap<Receiver>();
  if (!receivers) {
    return ToResultCode(receivers);
  }
  for (auto rx : receivers.value()) {
    if (!rx) {
      GXF_LOG_ERROR("Found bad queue in receivers");
      return GXF_FAILURE;
    }
    int64_t message_counter = 0;
    while (true) {
      // Honors message limit per tick
      if (tick_source_limit_ > 0 && message_counter >= tick_source_limit_) {
        break;
      }
      message_counter++;
      const auto message = rx.value()->receive();
      if (!message) {
        break;
      }
      const auto result = sink_->publish(message.value());
      if (!result) {
        return result.error();
      }
    }
  }
  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
