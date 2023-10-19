/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/std/synthetic_clock.hpp"

#include "gxf/std/gems/utils/time.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t SyntheticClock::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(initial_timestamp_, "initial_timestamp", "Initial Timestamp",
                                 "The initial timestamp on the clock (in nanoseconds).", 0L);
  return gxf::ToResultCode(result);
}

gxf_result_t SyntheticClock::initialize() {
  current_time_ = initial_timestamp_;
  return GXF_SUCCESS;
}

gxf_result_t SyntheticClock::deinitialize() {
  // wake up everybody one last time
  condition_variable_.notify_all();

  return GXF_SUCCESS;
}

double SyntheticClock::time() const {
  return gxf::TimestampToTime(timestamp());
}

int64_t SyntheticClock::timestamp() const {
  return current_time_;
}

gxf::Expected<void> SyntheticClock::sleepFor(int64_t duration_ns) {
  return sleepUntil(current_time_ + duration_ns);
}

gxf::Expected<void> SyntheticClock::sleepUntil(int64_t target_time_ns) {
  std::unique_lock<std::mutex> mutex_lock(mutex_);

  condition_variable_.wait(mutex_lock, [this, target_time_ns] {
    return current_time_ >= target_time_ns;
  });

  return gxf::Success;
}

gxf::Expected<void> SyntheticClock::advanceTo(int64_t new_time_ns) {
  std::lock_guard<std::mutex> mutex_lock(mutex_);

  current_time_ = new_time_ns;

  condition_variable_.notify_all();

  return gxf::Success;
}

gxf::Expected<void> SyntheticClock::advanceBy(int64_t time_delta_ns) {
  return advanceTo(current_time_ + time_delta_ns);
}

}  // namespace gxf
}  // namespace nvidia
