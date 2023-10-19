/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/behavior_tree/switch_behavior.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t SwitchBehavior::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(children_, "children",
                                 "Child Entities' BT Scheduling Terms",
                                 "Used to schedule child entities"
                                 "start/stop child entities");
  result &= registrar->parameter(s_term_, "s_term",
                                 "scheduling term for the entity itself",
                                 "Used to schedule the entity itself");
  result &= registrar->parameter(
      desired_behavior_, "desired_behavior",
      "the index of child entity to switch to when this entity runs");
  return ToResultCode(result);
}

gxf_result_t SwitchBehavior::initialize() {
  current_child_id = 0;
  children = children_.get();
  for (size_t i = 0; i < children.size(); i++) {
    children_eid.push_back(children.at(i).value()->eid());
    GXF_LOG_INFO("Parent %zu children %zu eid %zu", eid(), i, children_eid[i]);
  }
  s_term = s_term_.get();
  ready_conditions = SchedulingConditionType::READY;
  never_conditions = SchedulingConditionType::NEVER;

  return GXF_SUCCESS;
}

gxf_result_t SwitchBehavior::startChild(size_t child_id) {
  children.at(child_id).value()->set_condition(ready_conditions);
  return GXF_SUCCESS;
}

size_t SwitchBehavior::getNumChildren() const { return children.size(); }

entity_state_t SwitchBehavior::GetChildStatus(size_t child_id) {
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
  // parent codelet
  entity_state_t child_behavior_status;
  gxf_result_t result = GxfEntityGetState(context(), children_eid[child_id],
                                          &child_behavior_status);
  if (result != GXF_SUCCESS) return GXF_BEHAVIOR_UNKNOWN;
  return child_behavior_status;
}

gxf_result_t SwitchBehavior::tick() {
  // Get handle to parent entity
  auto this_entity = Entity::Shared(context(), eid());
  if (!this_entity) {
    return ToResultCode(this_entity);
  }

  if (isFirstTick()) {
    // In case there are no children the switch succeeds.
    if (getNumChildren() == 0) {
      GXF_LOG_INFO("'switch without children'");
      s_term->set_condition(never_conditions);
      return GXF_SUCCESS;
    }
    // Try to start the desired behavior
    current_child_id = desired_behavior_.get();
    startChild(current_child_id);
    return GXF_NOT_FINISHED;
  }

  entity_state_t child_status = GetChildStatus(current_child_id);
  // Get handle to child entity
  auto current_child_entity =
      Entity::Shared(context(), children_eid[current_child_id]);
  if (!current_child_entity) {
    return ToResultCode(current_child_entity);
  }

  switch (child_status) {
    case GXF_BEHAVIOR_SUCCESS:
      s_term->set_condition(never_conditions);
      return GXF_SUCCESS;
    case GXF_BEHAVIOR_FAILURE:
      s_term->set_condition(never_conditions);
      return GXF_FAILURE;
    case GXF_BEHAVIOR_INIT:
    case GXF_BEHAVIOR_RUNNING:
      s_term->set_condition(ready_conditions);
      return GXF_NOT_FINISHED;
    default:
      s_term->set_condition(never_conditions);
      return GXF_QUERY_NOT_FOUND;
  }
  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
