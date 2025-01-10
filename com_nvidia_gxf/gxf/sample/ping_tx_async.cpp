/*
Copyright (c) 2024 NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/sample/ping_tx_async.hpp"

#include <unistd.h>  // usleep

namespace nvidia {
namespace gxf {

gxf_result_t PingTxAsync::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(signal_, "signal", "Signal",
                                 "Transmitter channel publishing messages to other graph entities");
  result &= registrar->parameter(sleep_time_us_, "sleep_time_us", "Sleep Time (us)",
                                 "Time to sleep between sending messages", (int64_t)10);
  return ToResultCode(result);
}

gxf_result_t PingTxAsync::stop() {
  GXF_LOG_INFO("Sent %d unique messages", this->count);
  return GXF_SUCCESS;
}

gxf_result_t PingTxAsync::tick() {
  // usleep for certain (10 by default) us so that the sender runs at a different frequency compared
  // to the receiver. When the sender runs at a higher frequency than the receiver, it sends more
  // messages than the receiver has the opportunity to read. When the sender runs at a lower
  // frequency than the receiver, then the receiver may receive the same message more than once.
  // This is to demonstrate that the sender and receiver can run at different frequencies and
  // communicate asynchronously with a lock-free buffer.
  usleep(sleep_time_us_.get());
  auto message = Entity::New(context());
  if (!message) {
    GXF_LOG_ERROR("Failure creating message entity.");
    return message.error();
  }
  auto countcomponent = message.value().add<int>("count");
  this->count++;
  *countcomponent.value() = this->count;
  auto result = signal_->publish(message.value());
  GXF_LOG_INFO("Message Sent: %d", this->count);
  return ToResultCode(message);
}

}  // namespace gxf
}  // namespace nvidia
