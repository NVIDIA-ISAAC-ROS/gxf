/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_STD_PROGRAM_HPP_
#define NVIDIA_GXF_STD_PROGRAM_HPP_

#include <atomic>
#include <memory>
#include <string>
#include <unordered_set>

#include "common/fixed_vector.hpp"
#include "gxf/core/gxf.h"
#include "gxf/std/entity_executor.hpp"
#include "gxf/std/entity_warden.hpp"
#include "gxf/std/ipc_server.hpp"
#include "gxf/std/router_group.hpp"
#include "gxf/std/system.hpp"
#include "gxf/std/system_group.hpp"

namespace nvidia {
namespace gxf {

/// @brief Manages the program flow of a GXF application
///
/// RouterGroup & MessageRouter:
/// The program maintains one entity which holds a component of type RouterGroup. All router
/// components are added to the router group. The program also creates one component of type
/// MessageRouter which is used for standard message passing.
///
/// System:
/// The program maintains one entity which holds a component of type SystemGroup. All system
/// components are added to the system group.
class Program {
 public:
  // Origin ----------(activate)------------> Activating
  // Activating ----------------------------> Activated
  // Activated -------(run async)-----------> Starting
  // Starting ------------------------------> Running
  // Running ---------(interrupt)-----------> Interrupting
  // Running ---------(wait)----------------> Activated
  // Interrupting ----(wait)----------------> Activated
  // Activated -------(deactivate)----------> Deactivating
  // Deactivating --------------------------> Origin
  enum class State : int8_t {
    ORIGIN = 0,     // Not activated
    ACTIVATING,     // Executing activation sequence
    ACTIVATED,      // Activated, NOT running.
    STARTING,       // Activated and executing starting sequence.
    RUNNING,        // Activated and running.
    INTERRUPTING,   // Activated and executing stop sequence.
    DEINITALIZING,  // Executing deactivation sequence
  };

  Program();
  Expected<void> setup(gxf_context_t context, EntityWarden* warden, EntityExecutor* executor,
                       ParameterStorage* parameter_storage);
  Expected<void> addEntity(gxf_uid_t eid);
  Expected<void> scheduleEntity(gxf_uid_t eid);
  Expected<void> unscheduleEntity(gxf_uid_t eid);
  Expected<void> activate();
  Expected<void> runAsync();
  Expected<void> interrupt();
  Expected<void> wait();
  Expected<void> deactivate();
  Expected<void> destroy();
  Expected<void> entityEventNotify(gxf_uid_t eid);

 private:
  // pre activate actions on a list of entities that need to be done before activate
  Expected<void> preActivateEntities(const FixedVector<Entity, kMaxEntities>& entities);
  // pre deactivate actions on a list of entities that need to be done before deactivate
  Expected<void> preDeactivateEntities(const FixedVector<Entity, kMaxEntities>& entities);

  // activates a list of entities and deactivates the entire program if any entity
  // fails to activate
  Expected<void> activateEntities(FixedVector<Entity, kMaxEntities> entities);

  // dynamic parameter change
  Expected<void> onParameterSet(const std::string& resource, const std::string& data);
  // dump runtime graph information
  Expected<std::string> onGraphDump(const std::string& resource);
  // dump the who graph
  Expected<std::string> dumpGraph(gxf_uid_t uid);

  gxf_context_t context_;
  EntityWarden* entity_warden_;
  EntityExecutor* entity_executor_;

  std::atomic<State> state_;

  gxf_tid_t sys_tid_;

  Entity system_group_entity_;
  Handle<SystemGroup> system_group_;

  Entity router_group_entity_;
  Handle<RouterGroup> router_group_;

  std::recursive_mutex entity_mutex_;
  FixedVector<Entity> unscheduled_entities_;
  FixedVector<Entity> scheduled_entities_;
  std::unordered_set<gxf_uid_t> schedulers_;
  ParameterStorage* parameter_storage_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_STD_PROGRAM_HPP_
