/*
Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_STD_ENTITY_EXECUTOR_HPP_
#define NVIDIA_GXF_STD_ENTITY_EXECUTOR_HPP_

#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <utility>

#include "nvtx3/nvToolsExt.h"

#include "common/fixed_vector.hpp"
#include "common/logger.hpp"
#include "gxf/core/component.hpp"
#include "gxf/core/entity.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/std/clock.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/controller.hpp"
#include "gxf/std/job_statistics.hpp"
#include "gxf/std/message_router.hpp"
#include "gxf/std/monitor.hpp"
#include "gxf/std/network_router.hpp"
#include "gxf/std/router.hpp"
#include "gxf/std/scheduling_terms.hpp"
#include "gxf/std/system.hpp"

namespace nvidia {
namespace gxf {

#define GXF_STR_HELP(X) \
  case X:               \
    return #X;

// Executes entities.
class EntityExecutor {
 public:
  gxf_result_t initialize(Handle<Router> router,
                          Handle<MessageRouter> message_router,
                          Handle<NetworkRouter> network_router);

  gxf_result_t activate(gxf_context_t context, gxf_uid_t eid);

  gxf_result_t deactivate(gxf_uid_t eid);

  gxf_result_t deactivateAll();

  Expected<void> getEntities(FixedVectorBase<gxf_uid_t>& entities) const;

  // sets the clock in the router to be used to set pubtimestamps
  gxf_result_t setClock(Handle<Clock> clock);

  // Check if entity is ready to run and run it if possible. Returns scheduling conditions of the
  // entity anyway.
  Expected<SchedulingCondition> executeEntity(gxf_uid_t eid, int64_t timestamp);
  // Checks if entity is ready to run but not run it.
  Expected<SchedulingCondition> checkEntity(gxf_uid_t eid, int64_t timestamp);

  // Provides Statistics component to be updated when running codelets.
  Expected<void> addStatistics(Handle<JobStatistics> statistics);

  // Removes a statistics component from the executor
  Expected<void> removeStatistics(Handle<JobStatistics> statistics);

  // Adds a monitor to the executor
  Expected<void> addMonitor(Handle<Monitor> monitor);

  // Removes a monitor from the executor
  Expected<void> removeMonitor(Handle<Monitor> monitor);

  gxf_result_t getEntityStatus(gxf_uid_t eid, gxf_entity_status_t* entity_status);

  gxf_result_t getEntityBehaviorStatus(gxf_uid_t eid, entity_state_t& behavior_status);

  class EntityItem {
   public:
    const char* entityStatusStr(gxf_entity_status_t state);

    Entity entity;

    // store the EntityItem::execute() result for behavior parent codelet to query
    entity_state_t behavior_status = GXF_BEHAVIOR_INIT;

    // the controller for this entity
    Handle<Controller> controller;

    FixedVector<Handle<PeriodicSchedulingTerm>, kMaxComponents> periodic_terms;
    FixedVector<Handle<DownstreamReceptiveSchedulingTerm>, kMaxComponents>
        downstream_receptive_terms;
    FixedVector<Handle<SchedulingTerm>, kMaxComponents> terms;
    FixedVector<Handle<Codelet>, kMaxComponents> codelets;

    Expected<SchedulingCondition> check(int64_t timestamp) const;

    Expected<bool> activate(
        Entity other, MessageRouter* message_router,
        std::shared_ptr<FixedVector<Handle<JobStatistics>, kMaxComponents>> statistics,
        nvtxDomainHandle_t nvtx_domain, uint32_t nvtx_category_id);

    Expected<SchedulingCondition> execute(int64_t timestamp, Router* router,
                                          int64_t& ticking_variation);

    Expected<void> deactivate();

    Expected<gxf_entity_status_t> getEntityStatus();

   private:
    Expected<void> start(int64_t timestamp);

    Expected<void> tick(int64_t timestamp, Router* router);

    Expected<void> stop();

    Expected<void> startCodelet(const Handle<Codelet>& codelet);
    Expected<void> tickCodelet(const Handle<Codelet>& codelet);
    Expected<void> stopCodelet(const Handle<Codelet>& codelet);

    Expected<void> setEntityStatus(const gxf_entity_status_t next_state);

    mutable std::mutex mutex_;

    // Timestamp at which the entity was executed most recently
    int64_t last_execution_timestamp_;
    // The current status of the entity
    std::atomic<gxf_entity_status_t> status_;
    // Flag used to mark nvtx range for state changes
    bool first_status_change_ = true;
    // shared_ptr to statistics_ instance for collecting codelet metrics
    std::shared_ptr<FixedVector<Handle<JobStatistics>, kMaxComponents>> statistics_;

    // NVTX Domain handle used for profiling. All entities use one domain for GXF.
    nvtxDomainHandle_t nvtx_domain_;
    // NVTX Category id used for profiling. Each entity use a separate category.
    uint32_t nvtx_category_id_;
    // NVTX Range handles used for profiling
    nvtxRangeId_t nvtx_range_codelet_start_;
    nvtxRangeId_t nvtx_range_tick_codelet_;
    nvtxRangeId_t nvtx_range_codelet_stop_;
    nvtxRangeId_t nvtx_range_entity_state_;
  };

 private:
  mutable std::mutex mutex_;
  std::map<gxf_uid_t, std::unique_ptr<EntityItem>> items_;
  Handle<Router> router_;
  Handle<MessageRouter> message_router_;
  Handle<NetworkRouter> network_router_;

  mutable std::mutex statistics_mutex_;
  // List of statistics components that need to be updated when running entities
  std::shared_ptr<FixedVector<Handle<JobStatistics>, kMaxComponents>> statistics_;

  mutable std::mutex monitor_mutex_;
  // List of monitor components that perform a callback when an entity executes
  FixedVector<Handle<Monitor>, kMaxComponents> monitors_;

  // NVTX Domain handle used for profiling. All entities use one domain for GXF.
  nvtxDomainHandle_t nvtx_domain_;
  // The last NVTX Category id we have used for an entity
  uint32_t nvtx_category_id_last_ = 0;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_STD_ENTITY_EXECUTOR_HPP_
