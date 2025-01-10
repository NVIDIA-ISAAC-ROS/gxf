/*
Copyright (c) 2021,2023 NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/sample/ping_tx.hpp"

#include <chrono>
#include <thread>

namespace nvidia {
namespace gxf {

gxf_result_t PingTx::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(signal_, "signal", "Signal",
            "Transmitter channel publishing messages to other graph entities");
  result &= registrar->parameter(
      clock_, "clock", "Clock", "Clock component needed for timestamping messages",
      Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      trigger_interrupt_after_ms_, "trigger_interrupt_after_ms", "Trigger interrupt after ms",
      "Trigger interrupt after the specified time in milliseconds", Registrar::NoDefaultParameter(),
      GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->resource(gpu_device_, "Optional GPU device resource");
  return ToResultCode(result);
}

gxf_result_t PingTx::start() {
  // Create a thread that will trigger an interrupt after a specified time
  if (trigger_interrupt_after_ms_.try_get()) {
    std::thread([this]() {
      int64_t trigger_interrupt_after_ms = trigger_interrupt_after_ms_.try_get().value();
      GXF_LOG_INFO("Codelet [cid: %ld]: Sleeping for %ld ms", cid(), trigger_interrupt_after_ms);
      std::this_thread::sleep_for(std::chrono::milliseconds(trigger_interrupt_after_ms));
      GXF_LOG_INFO("Codelet [cid: %ld]: Calling interrupt", cid());
      GxfGraphInterrupt(context());
    }).detach();
  }

  if (gpu_device_.try_get()) {
    GXF_LOG_INFO("Codelet [cid: %ld]: GPUDevice value found and cached. "
                 "dev_id: %d", cid(), gpu_device_.try_get().value()->device_id());
  } else {
    GXF_LOG_DEBUG("Codelet [cid: %ld]: no GPUDevice found. "
                  "User need to prepare fallback case without GPU", cid());
  }
  return GXF_SUCCESS;
}

gxf_result_t PingTx::tick() {
  auto message = Entity::New(context());
  if (!message) {
    GXF_LOG_ERROR("Failure creating message entity.");
    return message.error();
  }
  auto maybe_clock = clock_.try_get();
  int64_t now;
  if (maybe_clock) {
    now = maybe_clock.value()->timestamp();
  } else {
    now = 0;
  }
  auto result = signal_->publish(message.value(), now);
  GXF_LOG_INFO("Message Sent: %d", this->count);
  this->count = this->count + 1;
  return ToResultCode(message);
}

}  // namespace gxf
}  // namespace nvidia
