/*
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/multi_thread_scheduler.hpp"

#include <algorithm>
#include <list>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gxf/core/gxf.h"
#include "gxf/core/registrar.hpp"
#include "gxf/std/entity_resource_helper.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t MultiThreadScheduler::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      clock_, "clock", "Clock",
      "The clock used by the scheduler to define flow of time. Typical choices are a RealtimeClock "
      "or a ManualClock.");
  result &= registrar->parameter(
      max_duration_ms_, "max_duration_ms", "Max Duration [ms]",
      "The maximum duration for which the scheduler will execute (in ms). If not specified the "
      "scheduler will run until all work is done. If periodic terms are present this means the "
      "application will run indefinitely.",
      Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      check_recession_period_ms_, "check_recession_period_ms",
      "Duration to sleep before checking the condition of an entity again [ms]",
      "The maximum duration for which the scheduler would wait (in ms) when an entity is not "
      "ready to run yet.",
      5.0);
  result &= registrar->parameter(
      stop_on_deadlock_, "stop_on_deadlock", "Stop on dead end",
      "If enabled the scheduler will stop when all entities are in a waiting state, but no "
      "periodic entity exists to break the dead end. Should be disabled when scheduling conditions "
      "can be changed by external actors, for example by clearing queues manually.",
      true);
  result &= registrar->parameter(worker_thread_number_, "worker_thread_number", "Thread Number",
                                 "Number of threads.", 1l);
  result &= registrar->parameter(
      thread_pool_allocation_auto_, "thread_pool_allocation_auto", "Automatic Pool Allocation",
      "If enabled, only one thread pool will be created. If disabled, user should enumerate pools"
      " and priorities",
      true);
  result &= registrar->parameter(
      strict_job_thread_pinning_, "strict_job_thread_pinning", "Strict Job-Thread Pinning",
      "If enabled, for entity pinned thread, it cannot execute other entities. "
      "i.e. true entity-thread pinning.",
      false);
  result &= registrar->parameter(
      stop_on_deadlock_timeout_, "stop_on_deadlock_timeout",
      "A refreshing version of max_duration_ms when stop_on_dealock kick-in [ms]",
      "Scheduler will wait this amount of time when stop_on_dead_lock indicates should stop. "
      "It will reset if a job comes in during the wait. Negative value means not stop on deadlock.",
      0l);
  return ToResultCode(result);
}

gxf_result_t MultiThreadScheduler::initialize() {
  {
    std::lock_guard<std::mutex> lock(conditions_mutex_);
    ready_count_ = 0;
    wait_time_count_ = 0;
    wait_event_count_ = 0;
    for (const auto pair : conditions_) {
      if (pair.second.type == SchedulingConditionType::READY) { ready_count_++; }
      if (pair.second.type == SchedulingConditionType::WAIT_TIME) { wait_time_count_++; }
      if (pair.second.type == SchedulingConditionType::WAIT_EVENT) { wait_event_count_++; }
    }
  }

  worker_jobs_ = std::make_unique<TimedJobList<gxf_uid_t>>(
      [this]() -> int64_t { return clock_.get()->timestamp(); });

  check_jobs_ = std::make_unique<TimedJobList<gxf_uid_t>>(
      [this]() -> int64_t { return clock_.get()->timestamp(); });

  event_waiting_ = std::make_unique<EventList<gxf_uid_t>>();
  event_notified_ = std::make_unique<EventList<gxf_uid_t>>();
  unschedule_entities_ = std::make_unique<EventList<gxf_uid_t>>();

  thread_error_code_ = GXF_SUCCESS;

  // populate internal default ThreadPool
  for (auto i = 0; i < worker_thread_number_; i++) {
    default_thread_pool_.addThread(i);
  }
  // add default thread pool as the first in the set
  thread_pool_set_.emplace(&default_thread_pool_);

  return GXF_SUCCESS;
}

void MultiThreadScheduler::updateCondition(gxf_uid_t eid,
                                           const SchedulingCondition& next_condition) {
  std::lock_guard<std::mutex> lock(conditions_mutex_);

  // New eid case, always mark as ready and gives to dispatcher thread
  if (conditions_.find(eid) == conditions_.end()) {
    conditions_[eid] = next_condition;
    ready_count_++;
    if (check_jobs_) { check_jobs_->insert(eid, clock_.get()->timestamp(), kMaxSlipNs, 0); }
    return;
  }

  // Existing eid case
  SchedulingCondition& prev_condition = conditions_[eid];
  if (next_condition.type != prev_condition.type) {
    if (prev_condition.type == SchedulingConditionType::READY) { ready_count_--; }
    if (next_condition.type == SchedulingConditionType::READY) { ready_count_++; }
    if (prev_condition.type == SchedulingConditionType::WAIT_TIME) { wait_time_count_--; }
    if (next_condition.type == SchedulingConditionType::WAIT_TIME) { wait_time_count_++; }
    if (prev_condition.type == SchedulingConditionType::WAIT_EVENT) { wait_event_count_--; }
    if (next_condition.type == SchedulingConditionType::WAIT_EVENT) { wait_event_count_++; }
    if (prev_condition.type == SchedulingConditionType::WAIT) { wait_count_--; }
    if (next_condition.type == SchedulingConditionType::WAIT) { wait_count_++; }
  }

  // entity is being unscheduled or end of execution lifecycle
  if (next_condition.type == SchedulingConditionType::NEVER) {
    conditions_.erase(conditions_.find(eid));
    return;
  }

  conditions_[eid] = next_condition;
}

void MultiThreadScheduler::dispatcherThreadEntrance() {
  while (state_ == State::kRunning) {
    gxf_uid_t eid = kNullUid;
    check_jobs_->waitForJob(eid);

    if (kNullUid == eid) {
      GXF_LOG_INFO("Dispatcher thread has no more jobs to check");
      break;
    }

    const int64_t now = clock_.get()->timestamp();
    Expected<SchedulingCondition> maybe_condition = \
                          SchedulingCondition{SchedulingConditionType::READY, now};
    gxf_entity_status_t entity_status;
    executor_->getEntityStatus(eid, &entity_status);
    if (unschedule_entities_->hasEvent(eid)) {
      unschedule_entities_->removeEvent(eid);
      maybe_condition = SchedulingCondition{SchedulingConditionType::NEVER, 0};
    } else if (entity_status != GXF_ENTITY_STATUS_NOT_STARTED) {
      // Checks the condition of entity
      maybe_condition = executor_->checkEntity(eid, now);
      if (!maybe_condition) {
        GXF_LOG_ERROR("Error while checking entity %zu: %s", eid,
                      GxfResultStr(maybe_condition.error()));
        // an error occurred, clean up and exit
        thread_error_code_ = maybe_condition.error();
        stopAllJobs();
        break;
      }
    }

    const SchedulingCondition& next_condition = maybe_condition.value();
    updateCondition(eid, next_condition);

    checkEndingCriteria(now);

    switch (next_condition.type) {
      case SchedulingConditionType::READY: {
        // Handles to worker threads
        worker_jobs_->insert(eid, now, kMaxSlipNs, 1);
      } break;
      case SchedulingConditionType::WAIT_EVENT: {
        event_waiting_->pushEvent(eid);
      } break;
      case SchedulingConditionType::WAIT_TIME: {
        // Handles to worker threads
        worker_jobs_->insert(eid, next_condition.target_timestamp, kMaxSlipNs, 1);
      } break;
      case SchedulingConditionType::WAIT: {
        // check it again later
        check_jobs_->insert(eid, now + check_recession_period_ms_ * kMsToNs, kMaxSlipNs, 0);
      } break;
      case SchedulingConditionType::NEVER: {
        // drops
      } break;
      default: {
        // an error occurred, clean up and exit
        GXF_LOG_ERROR("Unknown type of entity condition: %s",
                      SchedulingConditionTypeStr(next_condition.type));
        thread_error_code_ = GXF_ARGUMENT_OUT_OF_RANGE;
        stopAllJobs();
      } break;
    }
  }

  GXF_LOG_INFO("Dispatcher thread has stopped checking jobs");
  stopAllThreads();
  return;
}

bool MultiThreadScheduler::isJobMatch(ThreadPool* pool, int64_t thread_number,
                                      gxf_uid_t eid) {
  bool execute = false;
  if ((resources_.find(eid) == resources_.end())) {  // No entry means job is not pinned
    execute = true;
  } else {
    ThreadPool* job_pool = resources_[eid].first;
    int64_t job_thread = resources_[eid].second;
    // Matching pool and thread IDs means we are in the right worker thread
    if (job_pool == pool && job_thread == thread_number) {
      execute = true;
    }
    // else Mis-matching pool and thread IDs means we are in the wrong worker thread
  }
  return execute;
}

bool MultiThreadScheduler::isJobMatchStrict(ThreadPool* pool, int64_t thread_number,
                                            gxf_uid_t eid) {
  bool execute = false;
  auto it = resources_.find(eid);
  if (it == resources_.end()) {
    GXF_LOG_ERROR("Unscheduled entity eid: %ld, don't know which thread to execute it", eid);
    return false;
  }
  ThreadPool* job_pool = it->second.first;
  int64_t job_thread = it->second.second;
  // if this is default pool thread
  if (pool == &default_thread_pool_) {
    // If the job is assigned to a default pool
    if (job_pool == pool) {
      execute = true;
      GXF_LOG_DEBUG("Non-pinned job [eid: %ld] picked up by default pool "
                    "[ptr: %p, cid: %ld], random thread [uid: %ld]",
                    eid, pool, pool->cid(), thread_number);
    } else {
      // else wrong thread, not execute. Because the job is assigned to an added pool
      execute = false;
      GXF_LOG_DEBUG("Job [eid: %ld] skipped by default pool "
                    "[ptr: %p, cid: %ld], thread [uid: %ld]",
                    eid, pool, pool->cid(), thread_number);
    }
  // If this is an added pool thread
  } else {
    // If the job is assigned to this added pool and thread
    if (job_pool == pool && job_thread == thread_number) {
      execute = true;
      GXF_LOG_DEBUG("Pinned job [eid: %ld] picked up by matched pool [ptr: %p, cid: %ld], "
                    "thread [uid: %ld]", eid, pool, pool->cid(), thread_number);
    } else {
      // else wrong thread, not execute
      // because the job is either:
      // 1) assigned to another thread in the same added pool; or
      // 2) assigned to default pool
      execute = false;
      GXF_LOG_DEBUG("Job [eid: %ld] skipped by the mismatched pool "
                    "[ptr: %p, cid: %ld], thread[uid: %ld]",
                    eid, pool, pool->cid(), thread_number);
    }
  }
  return execute;
}

void MultiThreadScheduler::workerThreadEntrance(ThreadPool* pool, int64_t thread_number) {
  if (pool == nullptr) {
    GXF_LOG_ERROR("workerThreadEntrance has nullptr for arg ThreadPool*, exiting thread");
    return;
  }
  // Print thread param info
  std::string pool_name;
  if (pool == &default_thread_pool_) {
    pool_name = "default_pool";
  } else {
    pool_name = pool->name();
  }
  GXF_LOG_INFO("MultiThreadScheduler started worker thread [pool name: %s, thread uid: %ld]",
               pool_name.c_str(), thread_number);

  // Start thread loop
  while (true) {
    gxf_uid_t eid = kNullUid;
    worker_jobs_->waitForJob(eid);

    const char* entityName = "UNKNOWN";
    GxfParameterGetStr(context(), eid, kInternalNameParameterKey, &entityName);

    if (kNullUid == eid) {
      GXF_LOG_INFO("Worker Thread [pool name: %s, thread uid: %ld] exiting.",
                   pool_name.c_str(), thread_number);
      return;
    }

    // Do not execute if we have a pending request to unschedule
    if (unschedule_entities_->hasEvent(eid)) {
      unschedule_entities_->removeEvent(eid);
      updateCondition(eid, {SchedulingConditionType::NEVER, 0});
      continue;
    }

    bool execute;
    if (strict_job_thread_pinning_ == false) {
      execute = isJobMatch(pool, thread_number, eid);
    } else {
      execute = isJobMatchStrict(pool, thread_number, eid);
    }

    // Execute
    if (execute) {
      const int64_t now = clock_.get()->timestamp();
      const Expected<SchedulingCondition> maybe_condition = executor_->executeEntity(eid, now);
      if (!maybe_condition) {
        auto maybeEntity = nvidia::gxf::Entity::Shared(context(), eid);
        const char* entityName = "UNKNOWN";
        GxfParameterGetStr(context(), eid, kInternalNameParameterKey, &entityName);
        GXF_LOG_WARNING("Error while executing entity E%zu named '%s': %s", eid, entityName,
                        GxfResultStr(maybe_condition.error()));
        // an error occurred
        thread_error_code_ = maybe_condition.error();
        stopAllJobs();
        return;
      }
    }

    // Passes to dispatcher thread if scheduler is running
    // else sends a notification to the dispatcher thread indicating job done
    if (state_ == State::kRunning) {
      check_jobs_->insert(eid, clock_.get()->timestamp(), kMaxSlipNs, 0);
    } else {
      std::unique_lock<std::mutex> lock(state_change_mutex_);
      work_done_cv_.notify_one();
    }
  }
}

void MultiThreadScheduler::asyncEventThreadEntrance() {
  while (true) {
    if (state_ != State::kRunning) {
      GXF_LOG_INFO("Event handler thread exiting.");
      return;
    }

    // Check if we have received any event notifications
    std::list<gxf_uid_t> notifications = event_notified_->exportList();
    while (!notifications.empty()) {
      gxf_uid_t event = notifications.front();
      notifications.pop_front();

      // Check if entity was indeed waiting for event done notification
      if (event_waiting_->hasEvent(event)) {
        event_waiting_->removeEvent(event);
        event_notified_->removeEvent(event);
        const int64_t now = clock_.get()->timestamp();
        // Pass the entity to dispatcher thread
        check_jobs_->insert(event, now, kMaxSlipNs, 0);
      }
    }

    // wait until the next event notification
    std::unique_lock<std::mutex> lock(event_notification_mutex_);
    event_notification_cv_.wait(
        lock, [&] { return !event_notified_->empty() || state_ != State::kRunning; });
  }
}

void MultiThreadScheduler::checkEndingCriteria(int64_t timestamp) {
  // Checks if nothing to run (ready and wait_time jobs) and shall stop
  bool should_stop = false;
  {  // start of lock
    std::lock_guard<std::mutex> lock(conditions_mutex_);
    // only print status counts when users enable stope on dealock timeout
    if (stop_on_deadlock_timeout_.get() > 0) {
      GXF_LOG_DEBUG("ready_count_: %ld, wait_time_count_: %ld, wait_event_count_: %ld, "
        "wait_count_: %ld", ready_count_, wait_time_count_, wait_event_count_,
        wait_count_);
    }
    should_stop =
        stop_on_deadlock_ && ready_count_ == 0 && wait_time_count_ == 0 && wait_event_count_ == 0;

    // check if need to timeout the stop
    gxf_result_t result = stop_on_deadlock_timeout(stop_on_deadlock_timeout_.get(),
                          clock_.get()->timestamp(), should_stop);
    if (result != GXF_SUCCESS) {
      GXF_LOG_ERROR("Failed to re-evaluate should_stop based on timeout");
    }

    // check if need to force exit when all entities removed from dispatch queue
    if (stop_on_deadlock_ && ready_count_ == 0 && wait_time_count_ == 0 && wait_event_count_ == 0
        && wait_count_ == 0) {
      GXF_LOG_INFO("No entities left to schedule, force stopping");
      should_stop = true;
    }
  }  // end of lock
  if (should_stop) {
    GXF_LOG_INFO("No ready, wait time or wait event jobs. Exiting.");
    state_ = State::kStopping;

    // Check if any pending jobs can be executed
    while (!check_jobs_->empty()) {
      auto eid = check_jobs_->popFront();
      if (!eid) { break; }

      // Checks the condition of entity
      const Expected<SchedulingCondition> maybe_condition =
          executor_->checkEntity(eid.value(), timestamp);
      if (!maybe_condition) {
        GXF_LOG_ERROR("Error while checking entity %zu: %s", eid.value(),
                      GxfResultStr(maybe_condition.error()));
        // an error occurred, clean up and exit
        thread_error_code_ = maybe_condition.error();
        return;
      }

      if (maybe_condition.value().type == SchedulingConditionType::READY) {
        const int64_t now = clock_.get()->timestamp();
        worker_jobs_->insert(eid.value(), now, kMaxSlipNs, 1);
      }
    }

    std::unique_lock<std::mutex> lock(state_change_mutex_);
    work_done_cv_.wait(lock, [&] { return worker_jobs_->empty(); });
    stopAllJobs();
    return;
  }

  // Checks if duration expired
  const int64_t now = clock_.get()->timestamp();
  auto max_duration = max_duration_ms_.try_get();
  if (max_duration && now - start_timestamp_ > (*max_duration) * 1'000'000l) {
    GXF_LOG_INFO("Max duration expired. Exiting.");
    stopAllJobs();
    return;
  }
}

gxf_result_t MultiThreadScheduler::deinitialize() {
  async_threads_.clear();
  thread_pool_set_.clear();

  {
    std::lock_guard<std::mutex> lock(conditions_mutex_);
    conditions_.clear();
    ready_count_ = 0;
    wait_time_count_ = 0;
    wait_event_count_ = 0;
  }

  worker_jobs_.reset(nullptr);
  check_jobs_.reset(nullptr);
  event_waiting_.reset(nullptr);
  event_notified_.reset(nullptr);


  return thread_error_code_;
}

gxf_result_t MultiThreadScheduler::prepare_abi(EntityExecutor* executor) {
  executor_ = executor;
  return GXF_SUCCESS;
}

void MultiThreadScheduler::prepareResourceMap(gxf_uid_t eid) {
  Expected<Handle<ThreadPool>>
    entity_thread_pool = EntityResourceHelper::updateAndGetThreadPool(context(), eid);
  if (!entity_thread_pool && ToResultCode(entity_thread_pool) == GXF_RESOURCE_NOT_INITIALIZED) {
    GXF_LOG_DEBUG("Entity [eid: %05zu] is not prepared with pinned thread", eid);
    return;
  } else if (!entity_thread_pool) {
    GXF_LOG_ERROR("Failed to prepare thread for entity [eid: %05zu]", eid);
    return;
  }
  // If effectively pinned entity, i.e. has thread pool associated AND the pool has a thread for it
  if (entity_thread_pool && entity_thread_pool.value()->getThread(eid)) {
    resources_.emplace(eid,
                       std::make_pair(entity_thread_pool.value().get(),
                                      entity_thread_pool.value()->getThread(eid)->uid));
    thread_pool_set_.emplace(entity_thread_pool.value().get());
  }
}

void MultiThreadScheduler::prepareResourceMapStrict(gxf_uid_t eid) {
  Expected<Handle<ThreadPool>>
    entity_thread_pool = EntityResourceHelper::updateAndGetThreadPool(context(), eid);
  if (!entity_thread_pool && ToResultCode(entity_thread_pool) == GXF_RESOURCE_NOT_INITIALIZED) {
    GXF_LOG_DEBUG("Entity [eid: %05zu] is not prepared with pinned thread", eid);
    // DO NOT return
  } else if (!entity_thread_pool) {
    GXF_LOG_ERROR("Failed to update and get ThreadPool for entity [eid: %05zu]", eid);
    // DO NOT return
  }
  // If effectively pinned entity, i.e. has thread pool associated AND the pool has a thread for it
  if (entity_thread_pool && entity_thread_pool.value()->getThread(eid)) {
    resources_.emplace(eid,
                       std::make_pair(entity_thread_pool.value().get(),
                                      entity_thread_pool.value()->getThread(eid)->uid));
    thread_pool_set_.emplace(entity_thread_pool.value().get());
  } else {  // if not effectively pinned, assign default MTS thread pool
    resources_.emplace(eid, std::make_pair(&default_thread_pool_, -1));
  }
}

gxf_result_t MultiThreadScheduler::schedule_abi(gxf_uid_t eid) {
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

  const int64_t now = clock_.get()->timestamp();
  auto ready_conditions = SchedulingCondition{SchedulingConditionType::READY, now};
  updateCondition(eid, ready_conditions);

  if (strict_job_thread_pinning_ == false) {
    prepareResourceMap(eid);
  } else {
    prepareResourceMapStrict(eid);
  }

  return GXF_SUCCESS;
}

gxf_result_t MultiThreadScheduler::unschedule_abi(gxf_uid_t eid) {
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

  unschedule_entities_->pushEvent(eid);
  return GXF_SUCCESS;
}

gxf_result_t MultiThreadScheduler::runAsync_abi() {
  Handle<Clock> clock_handle;
  if (const auto maybe_clock = clock_.try_get()) {
    clock_handle = maybe_clock.value();
  } else {
    GXF_LOG_ERROR("Clock parameter must be set");
    return GXF_ARGUMENT_INVALID;
  }
  executor_->setClock(clock_handle);

  if (!async_threads_.empty()) {
    GXF_LOG_ERROR("Could not start scheduler again.");
    return GXF_INVALID_LIFECYCLE_STAGE;
  }

  if (executor_ == nullptr) {
    GXF_LOG_ERROR("No EntityExecutor to be used to run jobs.");
    return GXF_ARGUMENT_NULL;
  }

  if (worker_thread_number_.get() < 1) {
    GXF_LOG_ERROR("Must have at least 1 worker thread.");
    return GXF_PARAMETER_OUT_OF_RANGE;
  }
  async_threads_.reserve(worker_thread_number_.get() + 1);

  start_timestamp_ = clock_.get()->timestamp();

  state_ = State::kRunning;

  worker_jobs_->start();

  check_jobs_->start();

  // Creates the dispatcher thread
  dispatcher_thread_  = std::thread([this] { dispatcherThreadEntrance(); });

  // Creates async worker thread
  async_threads_.emplace_back([this] { asyncEventThreadEntrance(); });
  // Creates all worker threads
  for (const auto& thread_pool_ptr : thread_pool_set_) {
    for (const auto& thread_it : thread_pool_ptr->get()) {
      async_threads_.emplace_back([=] () { workerThreadEntrance(thread_pool_ptr,
                                                                thread_it.second.uid); });
    }
  }

  return GXF_SUCCESS;
}

gxf_result_t MultiThreadScheduler::stop_abi() {
  GXF_LOG_INFO("Stopping multithread scheduler");
  stopAllJobs();

  // dispatcher thread is responsible for cleaning up worker thread pool
  if (dispatcher_thread_.joinable()) { dispatcher_thread_.join(); }
  GXF_LOG_INFO("Multithread scheduler stopped.");
  return thread_error_code_;
}

gxf_result_t MultiThreadScheduler::wait_abi() {
  // sync work threads under lock since dispatcher can also sync them after execution is complete
  {
    std::unique_lock<std::mutex> lock(thread_sync_mutex_);
    thread_sync_cv_.wait(lock, [&] { return state_ != State::kRunning; });
    for (auto& thread : async_threads_) {
      if (thread.joinable()) { thread.join(); }
    }
  }

  if (dispatcher_thread_.joinable()) { dispatcher_thread_.join(); }
  GXF_LOG_INFO("Multithread scheduler finished.");
  return thread_error_code_;
}

gxf_result_t MultiThreadScheduler::event_notify_abi(gxf_uid_t eid) {
  GXF_LOG_DEBUG("Received event done notification for entity %ld", eid);
  std::unique_lock<std::mutex> lock(event_notification_mutex_);
  event_notified_->pushEvent(eid);
  event_notification_cv_.notify_one();
  return GXF_SUCCESS;
}

// This function must not be called from any of the async threads since we join them here
gxf_result_t MultiThreadScheduler::stopAllThreads() {
  GXF_LOG_INFO("Waiting to join all async threads");
  {
    std::unique_lock<std::mutex> lock(thread_sync_mutex_);
    for (auto& thread : async_threads_) {
      if (thread.joinable()) { thread.join(); }
    }
  }

  thread_sync_cv_.notify_all();
  GXF_LOG_INFO("All async worker threads joined, deactivating all entities");
  state_ = State::kStopped;
  return executor_->deactivateAll();  // Stop all the entities
}

void MultiThreadScheduler::stopAllJobs() {
  GXF_LOG_INFO("Stopping all async jobs");
  state_ = State::kStopping;
  check_jobs_->stop();                  // Stops dispatch thread
  worker_jobs_->stop();                 // Stops worker threads
  event_notified_->clear();             // Clear the event done list
  event_waiting_->clear();              // Clear the event waiting list
  unschedule_entities_->clear();        // Clear unscheduling requests
  event_notification_cv_.notify_one();  // Stops the async worker thread

  return;
}

gxf_result_t MultiThreadScheduler::stop_on_deadlock_timeout(const int64_t timeout,
                                            const int64_t now, bool& should_stop) {
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

  // 2. If having trend to stop in this interation,
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
