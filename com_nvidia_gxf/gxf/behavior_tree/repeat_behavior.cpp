/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/behavior_tree/repeat_behavior.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t RepeatBehavior::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(children_, "children",
                                 "Child Entities' BT Scheduling Terms",
                                 "Used to schedule child entities"
                                 "start/stop child entities");
  result &=
      registrar->parameter(s_term_, "s_term", "Scheduling Term",
                           "Used to schedule/unschedule the entity itself");
  result &= registrar->parameter(repeat_after_failure_, "repeat_after_failure");
  return ToResultCode(result);
}

gxf_result_t RepeatBehavior::initialize() {
  children = children_.get();
  for (size_t i = 0; i < children.size(); i++) {
    children_eid.push_back(children[i]->eid());
  }

  s_term = s_term_.get();
  ready_conditions = SchedulingConditionType::READY;
  never_conditions = SchedulingConditionType::NEVER;
  return GXF_SUCCESS;
}

gxf_result_t RepeatBehavior::startChild(size_t child_id) {
  children[child_id]->set_condition(ready_conditions);
  return GXF_SUCCESS;
}

size_t RepeatBehavior::getNumChildren() const { return children.size(); }

entity_state_t RepeatBehavior::GetChildStatus(size_t child_id) {
  if (child_id >= getNumChildren()) {
    GXF_LOG_ERROR(
        "Querying Child Status Failed. Query Child Id %05zu > "
        "Num of Children %05zu",
        child_id, getNumChildren());
    return GXF_BEHAVIOR_UNKNOWN;
  }
  // Use CAPI to query the behavior status of child entity
  // Call the Entity Executor's getEntityBehaviorStatus()
  // Return the behavior status stored inside EntityItem to Executor to this
  // behavior parent codelet
  entity_state_t child_behavior_status;
  gxf_result_t result = GxfEntityGetState(context(), children_eid[child_id],
                                          &child_behavior_status);
  if (result != GXF_SUCCESS) return GXF_BEHAVIOR_UNKNOWN;
  return child_behavior_status;
}

gxf_result_t RepeatBehavior::tick() {
  // Get handle to this entity
  auto this_entity = Entity::Shared(context(), eid());
  if (!this_entity) {
    return ToResultCode(this_entity);
  }

  if (isFirstTick()) {
    if (getNumChildren() != 1) {
      GXF_LOG_INFO(
          "['repeat with more than 1 or 0 children'] Entity %5ld('%s') Repeat "
          "Behavior only works with exactly one child. Got %zu",
          eid(), this_entity->name(), getNumChildren());
      s_term->set_condition(never_conditions);
      return GXF_SUCCESS;
    }
    // Start the first child
    startChild(0);
    s_term->set_condition(ready_conditions);
    return GXF_NOT_FINISHED;
  }

  bool restart_child = false;
  entity_state_t child_status = GetChildStatus(0);
  // Get handle to child entity
  auto current_child_entity = Entity::Shared(context(), children_eid[0]);
  if (!current_child_entity) {
    return ToResultCode(current_child_entity);
  }

  gxf_result_t result = GXF_SUCCESS;

  switch (child_status) {
    case GXF_BEHAVIOR_SUCCESS:
      restart_child = true;
      result = GXF_NOT_FINISHED;
      break;
    case GXF_BEHAVIOR_FAILURE:
      if (repeat_after_failure_.get()) {
        restart_child = true;
        result = GXF_NOT_FINISHED;
      } else {
        result = GXF_FAILURE;
      }
      break;
    case GXF_BEHAVIOR_UNKNOWN:
    case GXF_BEHAVIOR_RUNNING:
    case GXF_BEHAVIOR_INIT:
      restart_child = true;
      result = GXF_NOT_FINISHED;
      break;
    default:
      s_term->set_condition(never_conditions);
      return GXF_QUERY_NOT_FOUND;
  }

  if (restart_child) {
    s_term->set_condition(ready_conditions);
    startChild(0);
  } else {
    s_term->set_condition(never_conditions);
  }
  return result;
}

}  // namespace gxf
}  // namespace nvidia
