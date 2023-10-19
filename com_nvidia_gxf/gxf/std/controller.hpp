/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_STD_CONTROLLER_HPP_
#define NVIDIA_GXF_STD_CONTROLLER_HPP_

#include "gxf/core/component.hpp"
#include "gxf/std/scheduling_condition.hpp"

namespace nvidia {
namespace gxf {

/// @brief The status returned by controller to executor deciding termination
/// policy
/// - GXF_EXECUTE_SUCCESS: executor resumes execution
/// - GXF_EXECUTE_FAILURE_REPEAT: codelet fails and executor repeatedly execute
/// this entity
/// - GXF_EXECUTE_FAILURE_DEACTIVATE: codelet fails and executor deactivate this
/// entity
/// - GXF_EXECUTE_FAILURE: codelet fails and executor deactivates all the
/// entities and stops the entire graph
typedef enum {
  GXF_EXECUTE_SUCCESS = 0,
  GXF_EXECUTE_FAILURE_REPEAT = 1,
  GXF_EXECUTE_FAILURE_DEACTIVATE = 2,
  GXF_EXECUTE_FAILURE = 3,
} gxf_execution_status_t;

// Type to represent codelet::tick() result for behavior tree parent and
// execution status for executor to use termination policy
struct gxf_controller_status_t {
  entity_state_t behavior_status;
  gxf_execution_status_t exec_status;
  gxf_controller_status_t(entity_state_t b_status,
                          gxf_execution_status_t e_status)
      : behavior_status(b_status), exec_status(e_status) {}
  gxf_controller_status_t() {
    behavior_status = GXF_BEHAVIOR_SUCCESS;
    exec_status = GXF_EXECUTE_SUCCESS;
  }
};

// Interface for controlling entity's termination policy and entity's execution
// status
class Controller : public Component {
 public:
  virtual ~Controller() = default;

  // Return a struct encapsulating the determined behavior status and execution
  // status by the controller given the result of codelet's tick()
  virtual gxf_controller_status_t control(gxf_uid_t eid,
                                          Expected<void> code) = 0;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_STD_CONTROLLER_HPP_
