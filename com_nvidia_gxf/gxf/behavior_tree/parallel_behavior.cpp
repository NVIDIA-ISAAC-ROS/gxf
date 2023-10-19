/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/behavior_tree/parallel_behavior.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t ParallelBehavior::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(children_, "children",
                                 "Child Entities' BT Scheduling Terms",
                                 "Used to schedule child entities"
                                 "start/stop child entities");
  result &= registrar->parameter(s_term_, "s_term",
                                 "scheduling term for the entity itself",
                                 "Used to schedule the entity itself");
  result &= registrar->parameter(
      success_threshold_, "success_threshold",
      "Number of successful children required for success."
      " -1 means all children must succeed for this node to succeed.");
  result &= registrar->parameter(
      failure_threshold_, "failure_threshold",
      "Number of failed children required for failure."
      " -1 means all children must fail for this node to fail.");
  return ToResultCode(result);
}

gxf_result_t ParallelBehavior::initialize() {
  children = children_.get();
  for (size_t i = 0; i < children.size(); i++) {
    children_eid.push_back(children[i]->eid());
  }
  s_term = s_term_.get();
  ready_conditions = SchedulingConditionType::READY;
  never_conditions = SchedulingConditionType::NEVER;

  return GXF_SUCCESS;
}

gxf_result_t ParallelBehavior::startChild(size_t child_id) {
  children[child_id]->set_condition(ready_conditions);
  return GXF_SUCCESS;
}

gxf_result_t ParallelBehavior::stopAllChild() {
  for (size_t i = 0; i < getNumChildren(); i++) {
    children[i]->set_condition(never_conditions);
  }
  return GXF_SUCCESS;
}

size_t ParallelBehavior::getNumChildren() const { return children.size(); }

entity_state_t ParallelBehavior::GetChildStatus(size_t child_id) {
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

gxf_result_t ParallelBehavior::tick() {
  // Get handle to parent entity
  auto this_entity = Entity::Shared(context(), eid());
  if (!this_entity) {
    return ToResultCode(this_entity);
  }
  const size_t num_children = getNumChildren();
  if (isFirstTick()) {
    GXF_LOG_INFO("'[#(children)]'Entity %05ld('%s') has %05zu children\n", eid(),
                 this_entity->name(), num_children);

    // In case there are no children the Parallel succeeds.
    if (getNumChildren() == 0) {
      s_term->set_condition(never_conditions);
      return GXF_SUCCESS;
    }

    // Start all children
    for (size_t i = 0; i < children.size(); i++) {
      startChild(i);
    }
    return GXF_NOT_FINISHED;
  }

  // Count status of children
  size_t num_success = 0;
  size_t num_failure = 0;
  for (size_t i = 0; i < num_children; i++) {
    entity_state_t child_status = GetChildStatus(i);
    // Get handle to child entity
    auto current_child_entity = Entity::Shared(context(), children_eid[i]);
    if (!current_child_entity) {
      return ToResultCode(current_child_entity);
    }
    switch (child_status) {
      case GXF_BEHAVIOR_INIT:
        break;
      case GXF_BEHAVIOR_SUCCESS:
        num_success++;
        break;
      case GXF_BEHAVIOR_FAILURE:
        num_failure++;
        break;
      case GXF_BEHAVIOR_RUNNING:
        // not interested
        break;
      default:
        s_term->set_condition(never_conditions);
        stopAllChild();
        GXF_LOG_ERROR("child with unknown behavior status");
        return GXF_QUERY_NOT_FOUND;
    }
  }

  // Check success condition
  const int success_threshold = static_cast<int>(success_threshold_.get());
  if (success_threshold < -1) {
    GXF_LOG_INFO("Parallel %s Invalid success threshold", this_entity->name());
    s_term->set_condition(never_conditions);
    stopAllChild();
    return GXF_ARGUMENT_OUT_OF_RANGE;
  }
  if ((success_threshold == -1 && num_success == num_children) ||
      (num_success >= static_cast<size_t>(success_threshold))) {
    GXF_LOG_INFO("Parallel %s succeeds", this_entity->name());
    s_term->set_condition(never_conditions);
    stopAllChild();
    return GXF_SUCCESS;
  }

  // Check failure condition
  const int failure_threshold = static_cast<int>(failure_threshold_.get());
  if (failure_threshold < -1) {
    GXF_LOG_ERROR("Parallel %s Invalid failure threshold", this_entity->name());
    s_term->set_condition(never_conditions);
    stopAllChild();
    return GXF_ARGUMENT_OUT_OF_RANGE;
  }

  if ((failure_threshold == -1 && num_failure == num_children) ||
      (num_failure >= static_cast<size_t>(failure_threshold))) {
    GXF_LOG_ERROR("Parallel %s failed", this_entity->name());
    s_term->set_condition(never_conditions);
    stopAllChild();
    return GXF_FAILURE;
  }
  return GXF_NOT_FINISHED;
}

}  // namespace gxf
}  // namespace nvidia
