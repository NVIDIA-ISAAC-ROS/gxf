/*
Copyright (c) 2021,2023 NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/epoch_scheduler.hpp"

#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/memory_utils.hpp"

namespace nvidia {
namespace gxf {

namespace {
constexpr float kMinimumRunMs = 0.0f;
constexpr uint64_t kMsToNsConvertor = 1'000'000L;
}  // namespace

gxf_result_t EpochScheduler::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      clock_, "clock", "Clock",
      "The clock used by the scheduler to check maximum time budget. "
      "Typical choice is a RealtimeClock.");
  return ToResultCode(result);
}

gxf_result_t EpochScheduler::initialize() {
  active_entities_.reserve(kMaxEntities);
  event_entities_.reserve(kMaxEntities);
  entity_conditions_.clear();
  stopping_ = true;
  return GXF_SUCCESS;
}

gxf_result_t EpochScheduler::deinitialize() {
  active_entities_.clear();
  event_entities_.clear();
  entity_conditions_.clear();

  return GXF_SUCCESS;
}

gxf_result_t EpochScheduler::prepare_abi(EntityExecutor* executor) {
  executor_ = executor;
  return GXF_SUCCESS;
}

gxf_result_t EpochScheduler::schedule_abi(gxf_uid_t eid) {
  Expected<Entity> entity = Entity::Shared(context(), eid);
  if (!entity) {
    return ToResultCode(entity);
  }
  auto codelets = entity->findAll<Codelet>();
  if (!codelets) {
    return ToResultCode(codelets);
  }
  if (codelets->empty()) {
    // Ignores entity without codelet
    return GXF_SUCCESS;
  }

  std::mutex* entity_mutex;
  {
    std::lock_guard<std::mutex> conditions_lock(conditions_mutex_);
    auto result = entity_conditions_.emplace(
        eid,
        EntityCondition{SchedulingCondition{SchedulingConditionType::READY, 0},
                        std::make_unique<std::mutex>()});
    if (result.second) {
      // No existing eid found
      const auto result = active_entities_.push_back(eid);
      if (!result) {
        GXF_LOG_ERROR("Fail to schedule E%05zu.", eid);
        return GXF_FAILURE;
      }
      return GXF_SUCCESS;
    }
    entity_mutex = result.first->second.entity_mutex.get();
  }
  // To avoid deadlock, unlock and lock again in the same locking order
  // (entity => entity_conditions_) as needed in run_epoch_abi()
  {
    std::lock_guard<std::mutex> entity_lock(*entity_mutex);
    std::lock_guard<std::mutex> conditions_lock(conditions_mutex_);
    auto& entity_condition = entity_conditions_[eid];
    if (entity_condition.condition.type != SchedulingConditionType::NEVER) {
      GXF_LOG_ERROR("E%05zu is already scheduled.", eid);
      return GXF_FAILURE;
    }
    entity_condition.condition.type = SchedulingConditionType::READY;
    const auto result = active_entities_.push_back(eid);
    if (!result) {
      GXF_LOG_ERROR("Fail to schedule E%05zu.", eid);
      return GXF_FAILURE;
    }
  }

  return GXF_SUCCESS;
}

gxf_result_t EpochScheduler::unschedule_abi(gxf_uid_t eid) {
  Expected<Entity> entity = Entity::Shared(context(), eid);
  if (!entity) {
    return ToResultCode(entity);
  }
  auto codelets = entity->findAll<Codelet>();
  if (!codelets) {
    return ToResultCode(codelets);
  }
  if (codelets->empty()) {
    // Ignores entity without codelet
    return GXF_SUCCESS;
  }

  std::mutex* entity_mutex;
  {
    std::lock_guard<std::mutex> conditions_lock(conditions_mutex_);
    std::unordered_map<gxf_uid_t, EntityCondition>::iterator it =
        entity_conditions_.find(eid);
    if (it == entity_conditions_.end()) {
      GXF_LOG_ERROR("E%05zu is not scheduled yet.", eid);
      return GXF_FAILURE;
    }
    entity_mutex = it->second.entity_mutex.get();
  }
  // To avoid deadlock, unlock and lock again in the same locking order
  // (entity => entity_conditions_) as needed in run_epoch_abi()
  {
    std::lock_guard<std::mutex> entity_lock(*entity_mutex);
    std::lock_guard<std::mutex> conditions_lock(conditions_mutex_);
    entity_conditions_[eid].condition.type = SchedulingConditionType::NEVER;
  }

  return GXF_SUCCESS;
}

gxf_result_t EpochScheduler::runAsync_abi() {
  if (executor_ == nullptr) {
    return GXF_ARGUMENT_NULL;
  }

  // Get the clock used by the scheduler
  executor_->setClock(clock_.get());

  std::lock_guard<std::mutex> execution_lock(execution_mutex_);

  if (!stopping_) {
    GXF_LOG_INFO("Already started.");
    return GXF_SUCCESS;
  }
  stopping_ = false;

  return GXF_SUCCESS;
}

Expected<void> EpochScheduler::runEpoch(float budget_ms) {
  return ExpectedOrCode(run_epoch_abi(budget_ms));
}

gxf_result_t EpochScheduler::run_epoch_abi(float budget_ms) {
  std::lock_guard<std::mutex> execution_lock(execution_mutex_);

  // To stay within budget
  const int64_t clock_start = clock_.get()->timestamp();

  bool all_covered = false;  // Used in case only run all nodes once
  while (!stopping_) {
    const int64_t now = clock_.get()->timestamp();

    // Checks termination conditions
    if (budget_ms > kMinimumRunMs) {
      if (now >
          clock_start +
              budget_ms * kMsToNsConvertor) {  // convert duration from ms to ns
        GXF_LOG_INFO(
            "Epoch ending: time budget reached (clock_start=%ld, now=%ld, "
            "epoch_budget=%fms).",
            clock_start, now, budget_ms);
        return GXF_SUCCESS;
      }
      // Keeps running
    } else {
      // Only runs everything once if no budget is provided
      if (all_covered) {
        return GXF_SUCCESS;
      }
      all_covered = true;
    }

    // Handles event request
    auto event_result = processEventRequests();
    if (!event_result) {
      return ToResultCode(event_result);
    }

    if (active_entities_.empty()) {
      GXF_LOG_INFO("Epoch ending: Nothing to execute.");
      return GXF_SUCCESS;
    }

    // Indices of entities which are not supposed to be executed in the next
    // cycle
    FixedVector<size_t, kMaxEntities> to_remove_idx;

    // Check if entities are ready.
    size_t count_ready = 0;
    for (size_t i = 0; i < active_entities_.size(); ++i) {
      Expected<SchedulingCondition> maybe_condition = Unexpected{GXF_FAILURE};
      const gxf_uid_t eid = active_entities_.at(i).value();
      SchedulingCondition entity_condition;
      std::mutex* entity_mutex;
      {
        std::lock_guard<std::mutex> conditions_lock(conditions_mutex_);
        std::unordered_map<gxf_uid_t, EntityCondition>::iterator it =
            entity_conditions_.find(eid);
        entity_condition = it->second.condition;
        entity_mutex = it->second.entity_mutex.get();
      }

      {
        // Locks to avoid racing with schedule_abi()
        std::lock_guard<std::mutex> entity_lock(*entity_mutex);
        if (entity_condition.type == SchedulingConditionType::NEVER) {
          // unscheduled via unschedule_abi(). Stops ticking it.
          to_remove_idx.push_back(i);
          continue;
        }

        maybe_condition = executor_->executeEntity(eid, now);

        if (!maybe_condition) {
          const char* entityName = "UNKNOWN";
          GxfParameterGetStr(context(), eid, kInternalNameParameterKey,
                             &entityName);  // Ignore query error
          GXF_LOG_ERROR("Error while executing entity %zu named '%s': %s", eid,
                        entityName, GxfResultStr(maybe_condition.error()));
          return maybe_condition.error();
        }
        {
          std::lock_guard<std::mutex> conditions_lock(conditions_mutex_);
          entity_conditions_[eid].condition = maybe_condition.value();
        }
      }

      switch (maybe_condition.value().type) {
        case SchedulingConditionType::READY: {
          // run it again
          count_ready++;
        } break;
          // Both wait_time and wait cases need revisit
        case SchedulingConditionType::WAIT_TIME:
        case SchedulingConditionType::WAIT:
          break;
          // Neither wait_event and never case needs revisit
        case SchedulingConditionType::WAIT_EVENT:
        case SchedulingConditionType::NEVER: {
          to_remove_idx.push_back(i);
        } break;
      }
    }

    // Removes entities we shall not tick any more. Goes backwards to keep
    // indices valid. New eids from schedule_abi would be at the end of
    // active_entities_ and stay safe.
    {
      std::lock_guard<std::mutex> conditions_lock(conditions_mutex_);
      for (int32_t i = to_remove_idx.size() - 1; i >= 0; i--) {
        active_entities_.at(to_remove_idx[i].value()).value() =
            active_entities_.back().value();
        active_entities_.pop_back();
      }
    }

    {
      std::lock_guard<std::mutex> event_lock(event_mutex_);
      if (count_ready == 0 && event_entities_.empty()) {
        GXF_LOG_INFO("Epoch ending: No remaining entities to tick.");
        break;
      }
    }
  }

  // Pings waiting threads in wait_abi()
  execution_cv_.notify_all();
  return GXF_SUCCESS;
}

Expected<void> EpochScheduler::processEventRequests() {
  std::lock_guard<std::mutex> event_lock(event_mutex_);
  for (size_t i = 0; i < event_entities_.size(); ++i) {
    const gxf_uid_t eid = event_entities_[i].value();
    std::mutex* entity_mutex;
    {
      std::lock_guard<std::mutex> conditions_lock(conditions_mutex_);
      std::unordered_map<gxf_uid_t, EntityCondition>::iterator it =
          entity_conditions_.find(eid);
      if (it == entity_conditions_.end()) {
        GXF_LOG_ERROR("Signaled E%05zu is not scheduled yet.", eid);
        continue;
      }
      entity_mutex = it->second.entity_mutex.get();
    }
    // To avoid deadlock, unlock and lock again in the same locking order
    // (entity => entity_conditions_) as needed below for executeEntity()
    {
      std::lock_guard<std::mutex> entity_lock(*entity_mutex);
      std::lock_guard<std::mutex> conditions_lock(conditions_mutex_);
      std::unordered_map<gxf_uid_t, EntityCondition>::iterator it =
          entity_conditions_.find(eid);
      if (it->second.condition.type == SchedulingConditionType::WAIT_EVENT) {
        it->second.condition.type = SchedulingConditionType::READY;
        // Adds to active entities list to schedule
        const auto result = active_entities_.push_back(eid);
        if (!result) {
          GXF_LOG_ERROR("Error adding E%05ld to schedule", eid);
          return Unexpected{GXF_FAILURE};
        }
      }
    }
    // Ignores event if not waiting for event at all
  }
  event_entities_.clear();

  return Success;
}

gxf_result_t EpochScheduler::stop_abi() {
  if (stopping_) {
    GXF_LOG_INFO("Scheduler already stopping or stopped.");
  } else {
    GXF_LOG_INFO("Stopping scheduler.");
  }
  stopping_ = true;
  return GXF_SUCCESS;
}

gxf_result_t EpochScheduler::wait_abi() {
  std::unique_lock<std::mutex> execution_lock(execution_mutex_);
  if (!stopping_) {
    execution_cv_.wait(execution_lock, [&] { return stopping_; });
  }
  GXF_LOG_INFO("Scheduler finished.");
  return GXF_SUCCESS;
}

gxf_result_t EpochScheduler::event_notify_abi(gxf_uid_t eid) {
  std::unique_lock<std::mutex> lock(event_mutex_);
  const auto result = event_entities_.push_back(eid);
  if (!result) {
    GXF_LOG_ERROR("Error queuing event request for E%05ld", eid);
  }
  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
