/*
Copyright (c) 2020-2024 NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <algorithm>
#include <list>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "common/fixed_vector.hpp"
#include "common/memory_utils.hpp"
#include "gxf/std/entity_executor.hpp"
#include "gxf/std/greedy_scheduler.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t GreedyScheduler::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      clock_, "clock", "Clock",
      "The clock used by the scheduler to define flow of time. Typical choices are a RealtimeClock "
      "or a ManualClock.",
      Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      realtime_, "realtime", "Realtime (deprecated)",
      "This parameter is deprecated. Assign a clock directly.",
      Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      max_duration_ms_, "max_duration_ms", "Max Duration [ms]",
      "The maximum duration for which the scheduler will execute (in ms). If not specified the "
      "scheduler will run until all work is done. If periodic terms are present this means the "
      "application will run indefinitely.",
      Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      stop_on_deadlock_, "stop_on_deadlock", "Stop on dead end",
      "If enabled the scheduler will stop when all entities are in a waiting state, but no "
      "periodic entity exists to break the dead end. Should be disabled when scheduling conditions "
      "can be changed by external actors, for example by clearing queues manually.",
      true);
  result &= registrar->parameter(
      check_recession_period_ms_, "check_recession_period_ms",
      "Duration to sleep before checking the condition of next iteration [ms]",
      "The maximum duration for which the scheduler would wait (in ms) when "
      "all entities are not ready to run in current iteration.",
      0.0);
  result &= registrar->parameter(
      stop_on_deadlock_timeout_, "stop_on_deadlock_timeout",
      "A refreshing version of max_duration_ms when stop_on_dealock kick-in [ms]",
      "Scheduler will wait this amount of time when stop_on_dead_lock indicates should stop. "
      "It will reset if a job comes in during the wait. Negative value means not stop on deadlock.",
      0l);
  return ToResultCode(result);
}

gxf_result_t GreedyScheduler::initialize() {
  event_waiting_ = std::make_unique<EventList<gxf_uid_t>>();
  event_notified_ = std::make_unique<EventList<gxf_uid_t>>();
  unschedule_entities_ = std::make_unique<EventList<gxf_uid_t>>();
  active_entities_.reserve(kMaxEntities);
  new_entities_.reserve(kMaxEntities);
  return GXF_SUCCESS;
}

gxf_result_t GreedyScheduler::deinitialize() {
  thread_ = nullptr;
  clock_entity_ = Entity{};
  event_waiting_.reset(nullptr);
  event_notified_.reset(nullptr);
  unschedule_entities_.reset(nullptr);
  active_entities_.clear();
  new_entities_.clear();
  entity_mutex_map_.clear();

  return GXF_SUCCESS;
}

gxf_result_t GreedyScheduler::prepare_abi(EntityExecutor* executor) {
  executor_ = executor;
  return GXF_SUCCESS;
}

gxf_result_t GreedyScheduler::schedule_abi(gxf_uid_t eid) {
  Expected<Entity> entity = Entity::Shared(context(), eid);
  if (!entity) {
    return ToResultCode(entity);
  }
  auto codelets = entity->findAllHeap<Codelet>();
  if (!codelets) {
    return ToResultCode(codelets);
  }
  if (codelets->empty()) {
    // Ignores entity without codelet
    return GXF_SUCCESS;
  }

  std::lock_guard<std::mutex> lock(entity_mutex_);
  auto result = new_entities_.push_back(eid);
  if (!result) {
    GXF_LOG_WARNING("Exceeding container capacity");
    return GXF_EXCEEDING_PREALLOCATED_SIZE;
  }
  entity_mutex_map_.emplace(eid, std::make_unique<std::mutex>());

  return GXF_SUCCESS;
}

gxf_result_t GreedyScheduler::unschedule_abi(gxf_uid_t eid) {
  Expected<Entity> entity = Entity::Shared(context(), eid);
  if (!entity) {
    return ToResultCode(entity);
  }
  auto codelets = entity->findAllHeap<Codelet>();
  if (!codelets) {
    return ToResultCode(codelets);
  }
  if (codelets->empty()) {
    // Ignores entity without codelet
    return GXF_SUCCESS;
  }

  auto it = entity_mutex_map_.find(eid);
  if (it == entity_mutex_map_.end()) {
    // Entity has already been unscheduled
    return GXF_SUCCESS;
  }

  std::lock_guard<std::mutex> lock(*entity_mutex_map_.at(eid));
  unschedule_entities_->pushEvent(eid);

  return GXF_SUCCESS;
}

gxf_result_t GreedyScheduler::runAsync_abi() {
  if (executor_ == nullptr) { return GXF_ARGUMENT_NULL; }

  // Get the clock used by the scheduler
  Handle<Clock> clock_handle;
  if (const auto maybe_clock = clock_.try_get()) {
    clock_handle = maybe_clock.value();
  } else {
    const auto maybe_realtime = realtime_.try_get();
    if (!maybe_realtime) {
      GXF_LOG_ERROR("Clock parameter must be set");
      return GXF_ARGUMENT_INVALID;
    }
    GXF_LOG_WARNING("The deprecated parameter 'realtime_' is used. Set a clock directly.");
    // Create a clock
    auto maybe_entity = Entity::New(context());
    if (!maybe_entity) { return ToResultCode(maybe_entity); }
    clock_entity_ = std::move(maybe_entity.value());
    if (*maybe_realtime) {
      auto maybe = clock_entity_.add<RealtimeClock>();
      if (!maybe) { return ToResultCode(maybe); }
      clock_handle = maybe.value();
    } else {
      auto maybe = clock_entity_.add<ManualClock>();
      if (!maybe) { return ToResultCode(maybe); }
      clock_handle = maybe.value();
    }
    maybe_entity.value().activate();
  }

  executor_->setClock(clock_handle);

  // Start the execution thread
  thread_ = MakeUniqueNoThrow<std::thread>([this, clock_handle] {
    thread_error_code_ = GXF_SUCCESS;

    {
      // Load the entities parsed from the graph
      std::lock_guard<std::mutex> lock(entity_mutex_);
      for (size_t i = 0; i < new_entities_.size(); ++i) {
        active_entities_.emplace_back(new_entities_.at(i).value());
      }
      new_entities_.clear();
    }

    if (active_entities_.empty()) {
      GXF_LOG_WARNING("Nothing to schedule..");
      return;
    }
    GXF_LOG_INFO("Scheduling %zu entities", active_entities_.size());

    const int64_t start = clock_handle->timestamp();

    stopping_ = false;
    while (!stopping_) {
      const int64_t now = clock_handle->timestamp();

      // check termination conditions
      if (const auto max_duration = max_duration_ms_.try_get()) {
        const auto max_duration_ns = *max_duration * 1'000'000L;
        if (now > start + max_duration_ns) {
          GXF_LOG_WARNING("Scheduler stopped: Maximum duration reached (start=%ld, now=%ld, "
                          "max_duration=%ld).", start, now, max_duration_ns);
          thread_error_code_ = executor_->deactivateAll();  // Stop all the entities
          break;
        }
      }

      // Check if entities are ready. Step them if ready and remove them if not.
      // Handle runtime requests for scheduling, un-scheduling and async events.
      size_t count_ready = 0;
      Expected<int64_t> earliest_target = Unexpected{GXF_UNINITIALIZED_VALUE};

      {
        // handle any new scheduling requests
        std::lock_guard<std::mutex> lock(entity_mutex_);
        for (size_t i = 0; i < new_entities_.size(); ++i) {
          active_entities_.emplace_back(new_entities_.at(i).value());
        }
        new_entities_.clear();
      }

      // entities which are not supposed to be executed in the next cycle
      FixedVector<gxf_uid_t, kMaxEntities> to_remove;

      for (size_t i = 0; i < active_entities_.size(); ++i) {
        gxf_uid_t eid = active_entities_.at(i).value();

        Expected<SchedulingCondition> maybe_condition = Unexpected{GXF_FAILURE};
        {
          std::lock_guard<std::mutex> lock(*entity_mutex_map_.at(eid));
          if (unschedule_entities_->hasEvent(eid)) {
            maybe_condition = SchedulingCondition{SchedulingConditionType::NEVER, 0};
          } else {
            maybe_condition = executor_->executeEntity(eid, now);
          }
        }

        if (!maybe_condition) {
          const char* entityName = "UNKNOWN";
          GxfEntityGetName(context(), eid, &entityName);
          GXF_LOG_WARNING("Error while executing entity %zu named '%s': %s", eid, entityName,
                          GxfResultStr(maybe_condition.error()));
          // an error occurred
          thread_error_code_ = maybe_condition.error();
          executor_->deactivateAll();  // Stop all the entities
          return;
        }

        switch (maybe_condition.value().type) {
          case SchedulingConditionType::READY: {
            // run it again
            count_ready++;
          } break;
          case SchedulingConditionType::WAIT_TIME: {
            const int64_t target_timestamp = maybe_condition.value().target_timestamp;
            if (!earliest_target) { earliest_target = target_timestamp; }
            earliest_target = std::min(*earliest_target, target_timestamp);
          } break;
          case SchedulingConditionType::WAIT: {
            // run it again
            break;
          }
          case SchedulingConditionType::NEVER: {
            // unschedule forever
            auto result = to_remove.emplace_back(eid);
            if (!result) {
              thread_error_code_ = GXF_EXCEEDING_PREALLOCATED_SIZE;
              executor_->deactivateAll();  // Stop all the entities
              return;
            }
          } break;
          case SchedulingConditionType::WAIT_EVENT: {
            // unschedule for now, add it back once event done
            event_waiting_->pushEvent(eid);
            ++count_wait_event_;
            auto result = to_remove.emplace_back(eid);
            if (!result) {
              thread_error_code_ = GXF_EXCEEDING_PREALLOCATED_SIZE;
              executor_->deactivateAll();  // Stop all the entities
              return;
            }
          } break;
          default:
            break;
        }
      }

      // handle un-scheduling requests
      std::lock_guard<std::mutex> lock(entity_mutex_);
      while (!unschedule_entities_->empty()) {
        auto eid = unschedule_entities_->popEvent();
        for (size_t j = 0; j < active_entities_.size(); ++j) {
          if (active_entities_.at(j).value() == eid.value()) {
            active_entities_.erase(j);
            auto it = entity_mutex_map_.find(eid.value());
            entity_mutex_map_.erase(it);
          }
        }
      }

      // Check if we have received any event notifications
      std::list<gxf_uid_t> notifications = event_notified_->exportList();

      while (!notifications.empty()) {
        gxf_uid_t eid = notifications.front();
        notifications.pop_front();

       // Check if entity was indeed waiting for event done notification
        if (event_waiting_->hasEvent(eid)) {
          event_notified_->removeEvent(eid);
          event_waiting_->removeEvent(eid);
          // Pass the entity to be executed
          auto result = active_entities_.emplace_back(eid);
          if (!result) {
              thread_error_code_ = GXF_EXCEEDING_PREALLOCATED_SIZE;
              executor_->deactivateAll();  // Stop all the entities
              return;
          }
          --count_wait_event_;
          ++count_ready;
        }
      }

      // update the active entities
      for (size_t i = 0; i < to_remove.size(); ++i) {
        for (size_t j = 0; j < active_entities_.size(); ++j) {
          if (active_entities_.at(j).value() == to_remove.at(i).value()) {
            active_entities_.erase(j);
            break;
          }
        }
      }

      // check termination conditions
      if (active_entities_.empty() && event_waiting_->empty()) {
        GXF_LOG_INFO("Scheduler stopped: no more entities to schedule");
        break;
      }

      if (count_ready == 0) {
        if (earliest_target && *earliest_target > now) {
          // advance time so that the next periodic entity will tick immediately
          clock_handle->sleepFor(*earliest_target - now);
          // skip rest logic in block of count_ready = 0
          continue;
        } else if (count_wait_event_ > 0) {
          // If there are async entities waiting to be executed, sleep until the notification
          GXF_LOG_DEBUG("Waiting for an async event notification to run the next job");
          std::unique_lock<std::mutex> lock(event_mutex_);
          event_notification_cv_.wait(lock, [&]{ return !event_notified_->empty() || stopping_; });
          // skip rest logic in block of count_ready = 0
          continue;
        } else if (check_recession_period_ms_.get() > 0) {
          GXF_LOG_DEBUG("No READY entities in current iteration, sleep a recession period[%.2f ms] "
                        "before trying next iteration", check_recession_period_ms_.get());
          clock_handle->sleepFor(check_recession_period_ms_.get() * 1'000'000l);
        }

        // check stop on dealock, if not fall into above continue cases
        // co-exists with recession period
        if (stop_on_deadlock_) {
          bool should_stop = true;
          gxf_result_t result = stop_on_deadlock_timeout(stop_on_deadlock_timeout_.get(),
                                  now, should_stop);
          if (result != GXF_SUCCESS) {
            GXF_LOG_ERROR("Failed to re-evaluate should_stop based on timeout");
          }
          if (should_stop) {
            // If there are no periodic or async scheduling terms we are done
            GXF_LOG_INFO("Scheduler stopped: Some entities are waiting for execution, but there "
                        "are no periodic or async entities to get out of the deadlock.");
            break;
          }
        }
      } else {
        // update latest time when there are at least one ready count to execute
        // this is for stop_on_deadlock_timeout
        last_no_stop_ts_ = now;
      }
    }
  });
  if (!thread_) { return GXF_OUT_OF_MEMORY; }

  return GXF_SUCCESS;
}

gxf_result_t GreedyScheduler::stop_abi() {
  if (stopping_.exchange(true)) {
    GXF_LOG_INFO("Scheduler already stopping or stopped.");
  } else {
    GXF_LOG_INFO("Stopping scheduler.");
  }
  event_notification_cv_.notify_one();
  return GXF_SUCCESS;
}

gxf_result_t GreedyScheduler::wait_abi() {
  if (thread_) { thread_->join(); }
  GXF_LOG_INFO("Scheduler finished.");
  return thread_error_code_;
}

gxf_result_t GreedyScheduler::event_notify_abi(gxf_uid_t eid, gxf_event_t event) {
  GXF_LOG_DEBUG("Received event done notification for entity %ld", eid);
  if (event != GXF_EVENT_EXTERNAL) { return GXF_SUCCESS; }

  std::unique_lock<std::mutex> lock(event_mutex_);
  event_notified_->pushEvent(eid);
  event_notification_cv_.notify_one();
  return GXF_SUCCESS;
}

gxf_result_t GreedyScheduler::stop_on_deadlock_timeout(const int64_t timeout, const int64_t now,
                                bool& should_stop) {
  // only print debug when enable stop_on_deadlock_timeout, i.e. timeout > 0.
  // timeout = 0 is equivalent to stop_on_dealock without timeout
  if (timeout > 0) {
    GXF_LOG_DEBUG("timeout: %ld, now: %ld, last_no_stop_ts_:%ld, should_stop: %d",
                timeout, now, last_no_stop_ts_, should_stop);
  }

  // if timeout < 0, never stop on deadlock
  // should_stop = false & should_stop = false
  if (timeout < 0) {
    should_stop = false;
    // return immediately to skip step 2 and 3
    return GXF_SUCCESS;
  }

  // if timeout >= 0, enable stop on deadlock with a timeout.
  // Two steps:

  // 1. If no trend to stop in this iteration, update latest no-stop to now.
  // this is to maintain the latest time before first should_stop in a timeout period,
  // eg, TS_2 in below example
  // TS_1(!should_stop) -> TS_2(!SHOULD_STOP) -> TS_3(should_stop) -> TS_4(should_stop)
  if (!should_stop) {
    last_no_stop_ts_ = now;
    return GXF_SUCCESS;
  }

  // 2. If having trend to stop in this iteration,
  //     2.1 toggle should_stop to false if still within timeout period
  if (now - last_no_stop_ts_ < timeout * 1'000'000l) {
    GXF_LOG_DEBUG("Onhold trend to stop on deadlock for [%ld] ms",
                  (now - last_no_stop_ts_) / 1'000'000l);
    should_stop = false;
    // return with should_stop toggled to false
    return GXF_SUCCESS;

    // 3*. What's next. In subsequent timeout duration,
    //     3.1 if should_stop keeps to be true more timeout period,
    //         step 2.2 will let it return as true to trigger stop
    //     3.2 if should_stop had a chance to get re-evaluated to false by new wait counts
    //         during this timeout duration, only do step 1 to update latest no-stop time to now
  } else {
    // 2.2 do not toggle should_stop, leave should_stop as true to trigger stop.
    GXF_LOG_DEBUG("Agree to stop, as the trend to stop on deadlock retains over timeout period");
    // no coming back to this function anymore.
    return GXF_SUCCESS;
  }
}

}  // namespace gxf
}  // namespace nvidia
