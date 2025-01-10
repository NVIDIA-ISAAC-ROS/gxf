/*
Copyright (c) 2024 NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/sample/ping_rx_async.hpp"
#include <unistd.h>  // for usleep

namespace nvidia {
namespace gxf {

bool verify_integer_between_range(int value, int min, int max) {
  return value >= min && value <= max;
}

gxf_result_t PingRxAsync::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(signal_, "signal", "Signal",
                                 "Channel to receive messages from another graph entity");
  result &= registrar->parameter(sleep_time_us_, "sleep_time_us", "Sleep Time (us)",
                                 "Time to sleep between receiving messages", (int64_t)1000);
  result &= registrar->parameter(
      verification_range_min_, "verification_range_min", "Min value of verification range",
      "The minimum value of the range for verification of the received integers", (int32_t)0);
  result &= registrar->parameter(
      verification_range_max_, "verification_range_max", "Max value of verification range",
      "The maximum value of the range for verification of the received integers", (int32_t)500);
  return ToResultCode(result);
}

gxf_result_t PingRxAsync::stop() {
  GXF_LOG_INFO("Received %d unique messages", unique_received);
  return GXF_SUCCESS;
}

gxf_result_t PingRxAsync::tick() {
  // initialized to -1 because after the first message every message will have a positive value
  static int last_received = -1;
  // usleep for certain (1000 by default) us so that the receiver runs at a different frequency
  // compared to the the sender. When the receiver runs at a lower frequency than the sender, it
  // does not have the opportunity to read a few messages that are sent from the sender. When the
  // receiver runs at a higher frequency than the sender, it may receive the same message more than
  // once. This is to demonstrate that the sender and receiver can run at different frequencies and
  // communicate asynchronously with a lock-free buffer.
  usleep(sleep_time_us_.get());
  auto message = signal_->receive();
  if (!message || message.value().is_null()) {
    GXF_LOG_INFO("first message is not yet received.");
    return GXF_SUCCESS;
  }
  auto countcomponent = message.value().get<int>("count");
  GXF_LOG_INFO("Retrieved: %d", *countcomponent.value());
  if (last_received != *countcomponent.value()) {
    if (*countcomponent.value() < last_received) {
      GXF_LOG_ERROR("Received value is less than the last received value. Data integrity error.");
    }
    last_received = *countcomponent.value();
    if (!verify_integer_between_range(last_received, verification_range_min_.get(),
                                      verification_range_max_.get())) {
      GXF_LOG_ERROR("Received value is not in the expected range. Data integrity error.");
    }
    unique_received++;
  }
  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
