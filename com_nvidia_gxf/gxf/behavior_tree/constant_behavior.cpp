/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/behavior_tree/constant_behavior.hpp"
namespace nvidia {
namespace gxf {

gxf_result_t ConstantBehavior::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(s_term_, "s_term",
                                 "scheduling term for the entity itself",
                                 "Used to schedule the entity itself");
  result &= registrar->parameter(constant_status_, "constant_status");
  return ToResultCode(result);
}

gxf_result_t ConstantBehavior::initialize() {
  ready_conditions = SchedulingConditionType::READY;
  never_conditions = SchedulingConditionType::NEVER;

  return GXF_SUCCESS;
}

gxf_result_t ConstantBehavior::tick() {
  // Get handle to parent entity
  auto this_entity = Entity::Shared(context(), eid());
  if (!this_entity) {
    return ToResultCode(this_entity);
  }

  // switch to the desired status: constant_status_
  switch (constant_status_.get()) {
    case CONSTANT_SUCCESS:
      s_term_.get()->set_condition(never_conditions);
      return GXF_SUCCESS;
    case CONSTANT_FAILURE:
      s_term_.get()->set_condition(ready_conditions);
      return GXF_FAILURE;
    default:
      GXF_LOG_DEBUG(
          "[Unknown desired status of 'Constant Behavior'] @ eid [%6ld]'%s' -> "
          "Switch to status is invalid",
          eid(), this_entity->name());
      s_term_.get()->set_condition(ready_conditions);
      return GXF_FAILURE;
      break;
  }
  return GXF_NOT_FINISHED;
}

}  // namespace gxf
}  // namespace nvidia
