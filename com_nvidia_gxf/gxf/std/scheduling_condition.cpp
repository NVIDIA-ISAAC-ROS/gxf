/*
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <algorithm>

#include "gxf/core/gxf.h"
#include "gxf/std/scheduling_condition.hpp"

namespace nvidia {
namespace gxf {

SchedulingCondition AndCombine(SchedulingCondition a, SchedulingCondition b) {
  // "never" has the highest significance
  if (a.type == SchedulingConditionType::NEVER || b.type == SchedulingConditionType::NEVER) {
    return {SchedulingConditionType::NEVER, 0};
  }

  if (a.type == SchedulingConditionType::WAIT_EVENT ||
      b.type == SchedulingConditionType::WAIT_EVENT) {
    return {SchedulingConditionType::WAIT_EVENT, 0};
  }

  // "wait" is the second highest significance
  if (a.type == SchedulingConditionType::WAIT || b.type == SchedulingConditionType::WAIT) {
    return {SchedulingConditionType::WAIT, 0};
  }

  // "wait time" events are combined so that the maximum time is returned
  if (a.type == SchedulingConditionType::WAIT_TIME &&
      b.type == SchedulingConditionType::WAIT_TIME) {
    return {SchedulingConditionType::WAIT_TIME, std::max(a.target_timestamp, b.target_timestamp)};
  } else if (a.type == SchedulingConditionType::WAIT_TIME) {
    return a;
  } else if (b.type == SchedulingConditionType::WAIT_TIME) {
    return b;
  }

  // The only remaining case is that both are ready, choose the most recent timestamp to indicate
  // when the entity was finally ready
  return {SchedulingConditionType::READY, std::max(a.target_timestamp, b.target_timestamp)};
}

const char* SchedulingConditionTypeStr(const SchedulingConditionType& condition_type) {
  switch (condition_type) {
    GXF_ENUM_TO_STR(SchedulingConditionType::NEVER, Never)
    GXF_ENUM_TO_STR(SchedulingConditionType::READY, Ready)
    GXF_ENUM_TO_STR(SchedulingConditionType::WAIT, Wait)
    GXF_ENUM_TO_STR(SchedulingConditionType::WAIT_TIME, WaitTime)
    GXF_ENUM_TO_STR(SchedulingConditionType::WAIT_EVENT, WaitEvent)
    default:
      return "N/A";
  }
}

}  // namespace gxf
}  // namespace nvidia
