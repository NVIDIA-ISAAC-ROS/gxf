/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/timed_throttler.hpp"

#include <utility>

#include "gxf/std/gems/utils/time.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t TimedThrottler::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      transmitter_, "transmitter", "Transmitter",
      "Transmitter channel publishing messages at appropriate timesteps");
  result &= registrar->parameter(
      receiver_, "receiver", "Receiver",
      "Channel to receive messages that need to be synchronized");
  result &= registrar->parameter(
      execution_clock_, "execution_clock", "Execution Clock",
      "Clock on which the codelet is executed by the scheduler");
  result &= registrar->parameter(
      throttling_clock_, "throttling_clock", "Throttling Clock",
      "Clock on which the received entity timestamps are based");
  result &= registrar->parameter(
      scheduling_term_, "scheduling_term", "Scheduling Term",
      "Scheduling term for executing the codelet");
  return ToResultCode(result);
}

gxf_result_t TimedThrottler::initialize() {
  // Compute offset which is added to all timestamps
  time_offset_ = execution_clock_->timestamp() - throttling_clock_->timestamp();

  // Schedule an immediate execution to get the throttling started
  scheduling_term_->setNextTargetTime(execution_clock_->timestamp());

  cached_entity_ = Unexpected{GXF_UNINITIALIZED_VALUE};

  return GXF_SUCCESS;
}

gxf_result_t TimedThrottler::tick() {
  // Publish the cached entity from the previous tick
  if (cached_entity_) {
    auto maybe_publish = transmitter_->publish(std::move(*cached_entity_));
    cached_entity_ = Unexpected{GXF_UNINITIALIZED_VALUE};
    if (!maybe_publish) { return maybe_publish.error(); }
  }

  // Get the next message to be published and cache it
  auto maybe_next_message = receiver_->receive();
  if (!maybe_next_message) { return maybe_next_message.error(); }
  cached_entity_ = std::move(maybe_next_message.value());

  // Adapt timestamps
  auto maybe_next_timestamp = cached_entity_->get<Timestamp>();
  if (!maybe_next_timestamp) {
    return maybe_next_timestamp.error();
  }
  auto& next_timestamp = maybe_next_timestamp.value();
  next_timestamp->acqtime = next_timestamp->acqtime + time_offset_;
  next_timestamp->pubtime = next_timestamp->pubtime + time_offset_;

  // Update the scheduling term to tick at the desired target time
  scheduling_term_->setNextTargetTime(next_timestamp->acqtime);

  return GXF_SUCCESS;
}

gxf_result_t TimedThrottler::stop() {
  cached_entity_ = Unexpected{GXF_UNINITIALIZED_VALUE};
  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
