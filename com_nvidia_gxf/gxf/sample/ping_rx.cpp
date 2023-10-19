/*
Copyright (c) 2021,2023 NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/sample/ping_rx.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t PingRx::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(signal_, "signal", "Signal",
                      "Channel to receive messages from another graph entity");
  result &= registrar->resource(gpu_device_, "Optional GPU device resource");
  return ToResultCode(result);
}

gxf_result_t PingRx::start() {
  if (gpu_device_.try_get()) {
    GXF_LOG_INFO("Codelet [cid: %ld]: GPUDevice value found and cached. dev_id: %d",
      cid(), gpu_device_.try_get().value()->device_id());
  } else {
    GXF_LOG_DEBUG("Codelet [cid: %ld]: no GPUDevice found. "
      "User need to prepare fallback case without GPU", cid());
  }
  return GXF_SUCCESS;
}

gxf_result_t PingRx::tick() {
  auto message = signal_->receive();
  GXF_LOG_DEBUG("Message Received: %d", this->count);
  this->count = this->count + 1;
  if (!message || message.value().is_null()) {
    return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE;
  }
  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
