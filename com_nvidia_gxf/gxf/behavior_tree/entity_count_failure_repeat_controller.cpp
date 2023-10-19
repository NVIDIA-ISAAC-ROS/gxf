/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/behavior_tree/entity_count_failure_repeat_controller.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t EntityCountFailureRepeatController::registerInterface(
    Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      max_repeat_count_, "max_repeat_count", "Max repeat Count",
      "The max repeat count indicate the maximum number of repeating times "
      "after entity fails on execution"
      "The terimination policy for repeat controller is to repeat the entity "
      "if tick() fails "
      "until the repeating time reaches max_repeat_count");
  result &= registrar->parameter(return_behavior_running_if_failure_repeat_,
                                 "return_behavior_running_if_failure_repeat");
  return_behavior_running_if_failure_repeat_.set(false);
  return ToResultCode(result);
}

gxf_result_t EntityCountFailureRepeatController::initialize() {
  repeat_count_ = 0;
  return GXF_SUCCESS;
}

gxf_controller_status_t EntityCountFailureRepeatController::control(
    gxf_uid_t eid, Expected<void> code) {
  Expected<SchedulingCondition> expected_scheduling_condition =
      SchedulingCondition{SchedulingConditionType::READY, 0};
  if (!code) {
    expected_scheduling_condition = ForwardError(code);
  }
  // Set the behavior status based on gxf_result_t codelet::tick()
  controller_status.behavior_status =
      setBehaviorStatus(ToResultCode(expected_scheduling_condition));

  // Set the execution status based on behavior status and repeat_count
  controller_status.exec_status =
      setExecStatus(controller_status.behavior_status);
  return controller_status;
}

entity_state_t EntityCountFailureRepeatController::setBehaviorStatus(
    gxf_result_t tick_result) {
  switch (tick_result) {
    case GXF_SUCCESS:
      return GXF_BEHAVIOR_SUCCESS;
    case GXF_NOT_FINISHED:
      return GXF_BEHAVIOR_RUNNING;
    default:
      // for different types of Unexpected{error}
      GXF_LOG_INFO("Tick result: %s", GxfResultStr(tick_result));
      return GXF_BEHAVIOR_FAILURE;
  }
}

gxf_execution_status_t EntityCountFailureRepeatController::setExecStatus(
    entity_state_t& behavior_status) {
  // Get handle to this entity
  auto this_entity = Entity::Shared(context(), eid());
  if (!this_entity) {
    return GXF_EXECUTE_FAILURE;
  }
  switch (behavior_status) {
    case GXF_BEHAVIOR_INIT:
        return GXF_EXECUTE_FAILURE_REPEAT;
    case GXF_BEHAVIOR_RUNNING:
    case GXF_BEHAVIOR_SUCCESS:
      return GXF_EXECUTE_SUCCESS;
    case GXF_BEHAVIOR_FAILURE:
      repeat_count_++;
      GXF_LOG_INFO("Failure count is: %zu, max_repeat_count = %zu",
                   repeat_count_, max_repeat_count_.get());
      if (repeat_count_ <= max_repeat_count_.get()) {
        GXF_LOG_INFO("Controller Repeating Entity %s ['%zu/%zu']",
                     this_entity->name(), repeat_count_,
                     max_repeat_count_.get());

        if (return_behavior_running_if_failure_repeat_.get()) {
          behavior_status = GXF_BEHAVIOR_RUNNING;
        }
        return GXF_EXECUTE_FAILURE_REPEAT;
      } else {
        GXF_LOG_ERROR(
            "Entity %s Exceeding Controller Maximum Failure Repeating Count => %ld"
            "Will deactivate this entity",
            this_entity->name(), max_repeat_count_.get());
        return GXF_EXECUTE_FAILURE_DEACTIVATE;
      }
    default:
      return GXF_EXECUTE_SUCCESS;
  }
}

}  // namespace gxf
}  // namespace nvidia
