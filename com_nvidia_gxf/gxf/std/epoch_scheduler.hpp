/*
Copyright (c) 2021,2023 NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_STD_EPOCH_SCHEDULER_HPP_
#define NVIDIA_GXF_STD_EPOCH_SCHEDULER_HPP_

#include <algorithm>
#include <atomic>
#include <memory>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/fixed_vector.hpp"
#include "common/logger.hpp"
#include "gxf/core/component.hpp"
#include "gxf/core/entity.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/std/clock.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/entity_executor.hpp"
#include "gxf/std/gems/event_list/event_list.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/router.hpp"
#include "gxf/std/scheduler.hpp"
#include "gxf/std/scheduling_terms.hpp"
#include "gxf/std/system.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

/// @brief A scheduler intended to be run from a single external thread.

/// This scheduler is for running loads in externally managed threads. Each run
/// is called and Epoch. It would go over all entities that are known to be
/// active and execute them one by one. If epoch budget is provided (in ms), it
/// would keep running all codelets until budget is consumed or no codelet is
/// ready. It may run over budget as it guarantees to cover all codelets in
/// epoch. In case budget is not provided it would go over all codelets once and
/// execute them only once.

class EpochScheduler : public Scheduler {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override;
  gxf_result_t prepare_abi(EntityExecutor* executor) override;
  gxf_result_t schedule_abi(gxf_uid_t eid) override;
  gxf_result_t unschedule_abi(gxf_uid_t eid) override;
  gxf_result_t runAsync_abi() override;
  gxf_result_t stop_abi() override;
  gxf_result_t wait_abi() override;
  gxf_result_t event_notify_abi(gxf_uid_t eid, gxf_event_t event) override;

  // Runs the works with external thread. When budget is not positive, all nodes
  // are executed once.
  Expected<void> runEpoch(float budget_ms);

 private:
  // Handles queued event requests
  Expected<void> processEventRequests();

  gxf_result_t run_epoch_abi(float budget_ms);

  Parameter<Handle<Clock>> clock_;
  Parameter<bool> stop_on_deadlock_;
  EntityExecutor* executor_ = nullptr;
  bool stopping_ = true;

  // Guarding execution
  std::mutex execution_mutex_;
  std::condition_variable execution_cv_;

  struct EntityCondition {
    SchedulingCondition condition;
    // Guards access to condition above
    std::unique_ptr<std::mutex> entity_mutex;
  };

  // Used for keeping track of graph entities
  FixedVector<gxf_uid_t> active_entities_;
  std::unordered_map<gxf_uid_t, EntityCondition> entity_conditions_;
  // Guarding access to entity_conditions_ and active_entities_ above
  std::mutex conditions_mutex_;

  // Used for keeping track of async events
  std::mutex event_mutex_;
  FixedVector<gxf_uid_t> event_entities_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_STD_EPOCH_SCHEDULER_HPP_
