/*
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/sample/multi_ping_rx.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t MultiPingRx::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
            receivers_, "receivers", "Receivers",
            "A list of receivers which the entity can pop message entities");
  return ToResultCode(result);
}

gxf_result_t MultiPingRx::tick() {
  for (auto rx : receivers_.get()) {
    if (rx.value()->size()) {
      auto message = rx.value()->receive();
      GXF_LOG_INFO("Message Received at [%s]", rx.value()->name());
      if (!message || message.value().is_null()) {
        return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE;
      }
    }
  }
  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
