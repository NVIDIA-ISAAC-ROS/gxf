/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/clock.hpp"

#include <chrono>
#include <thread>

#include "gxf/std/gems/utils/time.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t RealtimeClock::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      initial_time_offset_, "initial_time_offset", "Initial Time Offset",
      "The initial time offset used until time scale is changed manually.", 0.0);
  result &= registrar->parameter(
      initial_time_scale_, "initial_time_scale", "Initial Time Scale",
      "The initial time scale used until time scale is changed manually.", 1.0);
  result &= registrar->parameter(
      use_time_since_epoch_, "use_time_since_epoch", "Use Time Since Epoch",
      "If true, clock time is time since epoch + initial_time_offset at initialize()."
      "Otherwise clock time is initial_time_offset at initialize().", false);
  return ToResultCode(result);
}

gxf_result_t RealtimeClock::initialize() {
  reference_ = std::chrono::steady_clock::now();

  time_offset_ = initial_time_offset_;
  if (use_time_since_epoch_.get()) {
    // We add the time_since_epoch to time_offset instead of reference, so that time_scale is
    // still correctly applied to the duration since reference time.
    time_offset_ += std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    GXF_LOG_INFO("Clock initial time %ld", TimeToTimestamp(time_offset_));
  }

  time_scale_ = initial_time_scale_;
  if (time_scale_ <= 0.0) {
    GXF_LOG_ERROR("Initial time scale cannot be negative %f", time_scale_);
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t RealtimeClock::deinitialize() {
  return GXF_SUCCESS;
}

double RealtimeClock::time() const {
  const auto now = std::chrono::steady_clock::now();
  const double delta =
      std::chrono::duration_cast<std::chrono::duration<double>>(now - reference_).count();

  return time_offset_ + time_scale_ * delta;
}

int64_t RealtimeClock::timestamp() const {
  return TimeToTimestamp(time());
}

Expected<void> RealtimeClock::sleepFor(int64_t duration_ns) {
  // Clock can not go backwards
  if (duration_ns < 0.0) {
    GXF_LOG_ERROR("Duration is negative: %ld. Clock cannot go backwards.", duration_ns);
    return Unexpected{GXF_FAILURE};
  }

  const int64_t delta = static_cast<int64_t>(static_cast<double>(duration_ns) / time_scale_);
  std::this_thread::sleep_for(std::chrono::duration<int64_t, std::nano>(delta));
  return Success;
}

Expected<void> RealtimeClock::sleepUntil(int64_t target_time_ns) {
  return sleepFor(target_time_ns - timestamp());
}

Expected<void> RealtimeClock::setTimeScale(double time_scale) {
  if (time_scale <= 0.0) {
    GXF_LOG_ERROR("Time scale cannot be negative: %f", time_scale);
    return Unexpected{GXF_FAILURE};
  }

  const auto now = std::chrono::steady_clock::now();
  const double delta =
      std::chrono::duration_cast<std::chrono::duration<double>>(now - reference_).count();

  reference_ = now;
  time_offset_ += time_scale_ * delta;
  time_scale_ = time_scale;

  return Success;
}

gxf_result_t ManualClock::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(initial_timestamp_, "initial_timestamp", "Initial Timestamp",
                                 "The initial timestamp on the clock (in nanoseconds).", 0L);
  return ToResultCode(result);
}

gxf_result_t ManualClock::initialize() {
  current_time_ = initial_timestamp_;
  return GXF_SUCCESS;
}

gxf_result_t ManualClock::deinitialize() {
  return GXF_SUCCESS;
}

double ManualClock::time() const {
  return TimestampToTime(timestamp());
}

int64_t ManualClock::timestamp() const {
  return current_time_;
}

Expected<void> ManualClock::sleepFor(int64_t duration_ns) {
  return sleepUntil(current_time_ + duration_ns);
}

Expected<void> ManualClock::sleepUntil(int64_t target_time_ns) {
  // Clock can not go backwards
  if (target_time_ns < current_time_) {
    GXF_LOG_ERROR("Target time %ld is less than current time %ld, Clock cannot go backwards",
                  target_time_ns, current_time_);
    return Unexpected{GXF_FAILURE};
  }
  current_time_ = target_time_ns;
  return Success;
}

}  // namespace gxf
}  // namespace nvidia
