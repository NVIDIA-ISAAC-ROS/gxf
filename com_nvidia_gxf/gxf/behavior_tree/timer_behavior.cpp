/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/behavior_tree/timer_behavior.hpp"
#include "gxf/std/gems/utils/time.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t TimerBehavior::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(s_term_, "s_term",
                                 "scheduling term for the entity itself",
                                 "Used to schedule the entity itself");
  result &= registrar->parameter(clock_, "clock", "Clock Component",
                                 "Used to keep track of time");
  result &= registrar->parameter(switch_status_, "switch_status");
  result &= registrar->parameter(delay_, "delay", "Time delay",
                                 "a specified time (in seconds) after which the status of this "
                                 "node will be changed");
  return ToResultCode(result);
}

gxf_result_t TimerBehavior::initialize() {
  is_first_tick_ = true;
  last_switch_timestamp_ = 0;
  return GXF_SUCCESS;
}

gxf_result_t TimerBehavior::tick() {
  const int64_t now = clock_.get()->timestamp();
  // Record the first start time stamp
  if (is_first_tick_) {
    last_switch_timestamp_ = now;
    is_first_tick_ = false;
  }

  // check if this_tick_time - first_tick_time is > delay_
  double cur_wait_time = TimestampToTime(now - last_switch_timestamp_);

  gxf_result_t result = GXF_NOT_FINISHED;
  if (cur_wait_time >= delay_.get()) {
    // switch to the desired behavior after "delay_"
    switch (switch_status_.get()) {
      case kSwitchToSuccess: {
        s_term_->set_condition(SchedulingConditionType::NEVER);
        result = GXF_SUCCESS;
      } break;
      case kSwitchToFailure: {
        s_term_->set_condition(SchedulingConditionType::NEVER);
        result = GXF_FAILURE;
      } break;
      case kSwitchToRunning: {
        s_term_->set_condition(SchedulingConditionType::READY);
        result = GXF_NOT_FINISHED;
      } break;
      default: {
        s_term_->set_condition(SchedulingConditionType::NEVER);
        result = GXF_FAILURE;
      } break;
    }
    // Update the timestamp on status switch
    last_switch_timestamp_ = now;
  } else {
    s_term_->set_condition(SchedulingConditionType::READY);
    result = GXF_NOT_FINISHED;
  }
  return result;
}

}  // namespace gxf
}  // namespace nvidia
