/*
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/event_based_scheduler.hpp"

#include <pthread.h>
#include <algorithm>
#include <chrono>
#include <list>
#include <memory>
#include <ratio>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gxf/core/gxf.h"
#include "gxf/core/registrar.hpp"
#include "gxf/std/entity_executor.hpp"
#include "gxf/std/entity_resource_helper.hpp"
#include "gxf/std/gems/utils/time.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t EventBasedScheduler::registerInterface(Registrar* registrar) {
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
      stop_on_deadlock_timeout_, "stop_on_deadlock_timeout",
      "A refreshing version of max_duration_ms when stop_on_dealock kick-in [ms]",
      "Scheduler will wait this amount of time when stop_on_dead_lock indicates should stop. "
      "It will reset if a job comes in during the wait. Negative value means not stop on deadlock.",
      0l);
  return ToResultCode(result);
}

gxf_result_t EventBasedScheduler::initialize() {
  //  first queue for the default thread pool
  ready_wait_time_jobs_.emplace_back(std::make_unique<TimedJobList<gxf_uid_t>>(
      [this]() -> int64_t { return clock_.get()->timestamp(); }));
  thread_queue_mapping_.reserve(kMaxThreads);

  event_waiting_ = std::make_unique<UniqueEventList<gxf_uid_t>>();
  waiting_ = std::make_unique<UniqueEventList<gxf_uid_t>>();
  external_event_notified_ = std::make_unique<UniqueEventList<gxf_uid_t>>();
  internal_event_notified_ = std::make_unique<UniqueEventList<gxf_uid_t>>();

  thread_error_code_ = GXF_SUCCESS;

  // populate internal default ThreadPool
  for (auto i = 0; i < worker_thread_number_; i++) {
    default_thread_pool_.addThread(i);
    // first queue for default thread pool
    thread_queue_mapping_.insert(std::make_pair(i, kDefaultThreadPoolQueueIndex));
  }
  // add default thread pool as the first in the set
  thread_pool_set_.emplace(&default_thread_pool_);

  return GXF_SUCCESS;
}

void EventBasedScheduler::updateCondition(std::shared_ptr<ScheduleEntity> e,
                                           const SchedulingCondition& next_condition) {
  if (!e) {
    GXF_LOG_ERROR("Received NULL entity");
    thread_error_code_ = GXF_NULL_POINTER;
    stopAllJobs();
    return;
  }
  auto prev_condition = e->condition_;

  // If previous condition was wait or wait event and if the next condition is different,
  // remove the eid from the event lists
  if (next_condition.type != prev_condition.type) {
    if (prev_condition.type == SchedulingConditionType::WAIT_EVENT) {
      event_waiting_->removeEvent(e->eid_);
    } else if (prev_condition.type == SchedulingConditionType::WAIT) {
      waiting_->removeEvent(e->eid_);
    }
  }

// PCG fix const error
  e->condition_ = next_condition;

  switch (next_condition.type) {
    case SchedulingConditionType::READY: {
      // Handles to worker threads
      std::unique_lock<std::shared_timed_mutex> lk_(e->ready_queue_sync_mutex_);
      ready_wait_time_jobs_[e->queue_index_]->insert(e->eid_, next_condition.target_timestamp,
       kMaxSlipNs, 1);
      e->is_present_in_ready_queue_ = true;
    } break;
    case SchedulingConditionType::WAIT_EVENT: {
      event_waiting_->pushEvent(e->eid_);
    } break;
    case SchedulingConditionType::WAIT_TIME: {
      // Moves to the ready_wait_time queue
      std::unique_lock<std::shared_timed_mutex> lk_(e->ready_queue_sync_mutex_);
      ready_wait_time_jobs_[e->queue_index_]->insert(e->eid_, next_condition.target_timestamp,
       kMaxSlipNs, 1);
      e->is_present_in_ready_queue_ = true;
    } break;
    case SchedulingConditionType::WAIT: {
      waiting_->pushEvent(e->eid_);
    } break;
    case SchedulingConditionType::NEVER: {
      // drops, remove any additional internal notifications
      internal_event_notified_->removeEvent(e->eid_);
      GXF_LOG_INFO("Unscheduling entity [%s] with id [%ld] from execution ",
                   e->name_.c_str(), e->eid_);
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

void EventBasedScheduler::dispatchEntityAsync(std::shared_ptr<ScheduleEntity> e) {
  dispatchEntity(e);
  const int64_t now = clock_.get()->timestamp();
  bool should_stop = checkEndingCriteria(now);
  if (should_stop) {
    GXF_LOG_WARNING("Deadlock detected after dispatch due to external event");
    notifyDispatcher();
  }
}

void EventBasedScheduler::dispatchEntity(std::shared_ptr<ScheduleEntity> e) {
  if (!e) {
    GXF_LOG_ERROR("Received NULL entity.");
    thread_error_code_ = GXF_NULL_POINTER;
    stopAllJobs();
    return;
  }

  const int64_t now = clock_.get()->timestamp();
  Expected<SchedulingCondition> maybe_condition = \
                        SchedulingCondition{SchedulingConditionType::READY, now};

  // Unschedule entity if needed, set condition to never
  if (e->unschedule_ == true) {
    maybe_condition = SchedulingCondition{SchedulingConditionType::NEVER, 0};
  } else {
    // Checks the condition of entity
    maybe_condition = executor_->checkEntity(e->eid_, now);
    if (!maybe_condition) {
      GXF_LOG_ERROR("Error while checking entity %ld: %s", e->eid_,
                    GxfResultStr(maybe_condition.error()));
      // an error occurred, clean up and exit
      thread_error_code_ = maybe_condition.error();
      stopAllJobs();
      return;
    }
  }

  // update the new condition
  const SchedulingCondition& next_condition = maybe_condition.value();
  GXF_LOG_VERBOSE("Entity [%s] scheduling condition [%s]\n", e->name_.c_str(),
                  SchedulingConditionTypeStr(next_condition.type));
  updateCondition(e, next_condition);
}

void EventBasedScheduler::dispatcherThreadEntrance() {
  pthread_setname_np(pthread_self(), "dispatcher");

  uint64_t dispatcher_end = getCurrentTimeUs();
  uint64_t  dispatcher_begin = getCurrentTimeUs();

  while (state_ == State::kRunning) {
    dispatcher_begin = getCurrentTimeUs();
    dispatcherStats.waitTime += (dispatcher_begin - dispatcher_end);
    dispatcherStats.execCount++;

    gxf_uid_t eid = kNullUid;

    // Check if we have received any internal event notifications
    while (!internal_event_notified_->empty()) {
      auto maybe_event = internal_event_notified_->popEvent();
      // TODO(pshekdar): error handling
      eid = maybe_event.value();
      std::shared_ptr<ScheduleEntity> e;
      if (kNullUid != eid)
        e = entities_[eid];

      // TODO(pshekdar): error handling
      if ((eid != kNullUid) &&
          (!executor_->isEntityBusy(eid))) {
            std::shared_lock<std::shared_timed_mutex> lk_(e->ready_queue_sync_mutex_);
            if (e->is_present_in_ready_queue_ == false) {
              lk_.unlock();
              dispatchEntity(e);
            }
      }
    }

    const int64_t now = clock_.get()->timestamp();
    bool should_stop = checkEndingCriteria(now);

    dispatcher_end = getCurrentTimeUs();
    dispatcherStats.execTime += (dispatcher_end - dispatcher_begin);

    if (should_stop) {
      GXF_LOG_VERBOSE("Dispatcher waiting for time %ld ms before checking for deadlock state",
                      stop_on_deadlock_timeout_.get());
      std::unique_lock<std::mutex> lock_internal_event(internal_event_notification_mutex_);
      bool wait_for_val = internal_event_notification_cv_.wait_for(
          lock_internal_event, std::chrono::milliseconds(stop_on_deadlock_timeout_.get()),
          [&] {
          return !internal_event_notified_->empty() || state_ != State::kRunning; });

      // If dispatcher wakes up due to timeout and not the predicate condition then check ending
      // criteria once again before stopping
      if (wait_for_val == false && checkEndingCriteria(clock_.get()->timestamp()) == true) {
        // condition variable woke up due to timeout and not some event
        GXF_LOG_WARNING("Deadlock! No ready, wait time or wait event jobs. Exiting.");
        state_ = State::kStopping;

        std::unique_lock<std::mutex> lock(state_change_mutex_);
        work_done_cv_.wait(lock, [&] { bool isEmpty = true;
          for (uint64_t i = 0; i < ready_wait_time_jobs_.size(); i++) {
            isEmpty = isEmpty && ready_wait_time_jobs_[i]->empty();
          }
          return isEmpty;
         });
        stopAllJobs();
      }
    } else {
      std::unique_lock<std::mutex> lock_internal_event(internal_event_notification_mutex_);
      internal_event_notification_cv_.wait(lock_internal_event,
          [&] { return !internal_event_notified_->empty() || state_ != State::kRunning; });
    }
  }

  GXF_LOG_INFO("Dispatcher thread has stopped checking jobs");
  stopAllThreads();
  return;
}

void EventBasedScheduler::workerThreadEntrance(ThreadPool* pool, int64_t thread_number) {
  auto thread_name = std::string("WorkerThread-") + std::to_string(thread_number);
  pthread_setname_np(pthread_self(), thread_name.c_str());
  if (pool == nullptr) {
    GXF_LOG_ERROR("workerThreadEntrance has nullptr for arg ThreadPool*, exiting thread");
    return;
  }
  // Print thread param info
  std::string pool_name = pool == &default_thread_pool_ ? "default_pool" : pool->name();

  GXF_LOG_INFO("EventBasedScheduler started worker thread [pool name: %s, thread uid: %ld]",
               pool_name.c_str(), thread_number);

  uint64_t worker_begin = getCurrentTimeUs();
  uint64_t worker_end = getCurrentTimeUs();

  // Start thread loop
  while (true) {
    gxf_uid_t eid = kNullUid;

    // Decrease the running count before going to sleep
    running_threads_.fetch_sub(1);
    ready_wait_time_jobs_[thread_queue_mapping_[thread_number].value()]->waitForJob(eid);
    if (kNullUid == eid) {
      GXF_LOG_VERBOSE("Worker Thread [pool name: %s, thread uid: %ld] exiting.",
                   pool_name.c_str(), thread_number);
      return;
    }

    std::shared_ptr<ScheduleEntity> e = entities_[eid];
    {
      std::unique_lock<std::shared_timed_mutex> lk_(e->ready_queue_sync_mutex_);
      e->is_present_in_ready_queue_ = false;
    }
    // Try to acquire / lock the entity for execution
    running_threads_.fetch_add(1);
    if (!e->tryToAcquire()) {
      // Entity has been acquired by another worker, drop this entity and proceed
      continue;
    }

    GXF_LOG_VERBOSE("Worker thread [%ld] acquired entity %s for execution \n",
                    thread_number, e->name_.c_str());
    worker_begin = getCurrentTimeUs();
    workerStats.waitTime += (worker_begin - worker_end);
    workerStats.execCount++;

    // Do not execute if we have a pending request to unschedule
    if (e->unschedule_ == true) {
      updateCondition(e, {SchedulingConditionType::NEVER, 0});
      continue;
    }

    const int64_t now = clock_.get()->timestamp();
    auto maybe_condition = executor_->executeEntity(eid, now);
    if (!maybe_condition) {
      // an error occurred
      GXF_LOG_WARNING("Error while executing entity E%zu named '%s': %s", eid,
                      e->name_.c_str(), GxfResultStr(maybe_condition.error()));
      thread_error_code_ = maybe_condition.error();
      stopAllJobs();
      notifyDispatcher();
      GXF_LOG_INFO("Worker thread with id [%ld] exiting", thread_number);
      running_threads_.fetch_sub(1);
      return;
    }

    e->releaseOwnership();

    // Passes to dispatcher thread if scheduler is running
    // else sends a notification to the dispatcher thread indicating job done
    if (state_ == State::kRunning) {
      notifyDispatcher(eid);
      worker_end = getCurrentTimeUs();
      workerStats.execTime += (worker_end - worker_begin);
    } else {
      std::unique_lock<std::mutex> lock(state_change_mutex_);
      work_done_cv_.notify_one();
      worker_end = getCurrentTimeUs();
      workerStats.execTime += (worker_end - worker_begin);
    }
  }
}

gxf_result_t EventBasedScheduler::notifyDispatcher(gxf_uid_t eid) {
  std::unique_lock<std::mutex> lock(internal_event_notification_mutex_);
  internal_event_notified_->pushEvent(eid);
  internal_event_notification_cv_.notify_one();
  return GXF_SUCCESS;
}

void EventBasedScheduler::asyncEventThreadEntrance() {
  pthread_setname_np(pthread_self(), "async");
  while (true) {
    if (state_ != State::kRunning) {
      GXF_LOG_INFO("Async event handler thread exiting.");
      return;
    }

    // Check if we have received any external event notifications
    while (!external_event_notified_->empty()) {
      auto maybe_event = external_event_notified_->popEvent();
      // TODO(pshekdar): error handling
      gxf_uid_t eid = maybe_event.value();
      std::shared_ptr<ScheduleEntity> e = entities_[eid];
        // Move the entity to appropriate queue
      GXF_LOG_DEBUG("Async event handler thread received event for entity [%s] with id [%ld]",
                    e->name_.c_str(), eid);
      if ((eid != kNullUid) && (!executor_->isEntityBusy(eid))) {
        std::shared_lock<std::shared_timed_mutex> lk_(e->ready_queue_sync_mutex_);
        if (e->is_present_in_ready_queue_ == false) {
          lk_.unlock();
          dispatchEntityAsync(e);
        }
      }
    }

    // wait until the next event notification
    std::unique_lock<std::mutex> lock(external_event_notification_mutex_);
    external_event_notification_cv_.wait(
        lock, [&] { return !external_event_notified_->empty() || state_ != State::kRunning; });
  }
}

uint64_t EventBasedScheduler::getReadyCount() {
  int readyCount = 0;
  for (uint64_t i = 0; i < ready_wait_time_jobs_.size(); i++) {
    readyCount += ready_wait_time_jobs_[i]->size();
  }
  return readyCount;
}

bool EventBasedScheduler::checkEndingCriteria(int64_t timestamp) {
  // Checks if nothing to run (ready and wait_time jobs) and shall stop
  bool should_stop = false;
  {
    // only print status counts when users enable stope on dealock timeout
    auto running_threads = running_threads_.load();
    if (stop_on_deadlock_timeout_.get() >= 0) {
      GXF_LOG_DEBUG("ready_wait_time_count: %ld, wait_event_count: %ld, "
        "wait_count: %ld, running jobs: %d", getReadyCount(), event_waiting_->size(),
        waiting_->size(), running_threads);
    }
    should_stop = state_ == State::kRunning && stop_on_deadlock_ &&
                            getReadyCount() == 0 && event_waiting_->size() == 0 &&
                            running_threads == 0;
  }  // end of lock
  return should_stop;
}

gxf_result_t EventBasedScheduler::deinitialize() {
  async_threads_.clear();
  thread_pool_set_.clear();

  for (uint64_t i = 0; i < ready_wait_time_jobs_.size(); i++) {
    ready_wait_time_jobs_[i].reset(nullptr);
  }
  event_waiting_.reset(nullptr);
  external_event_notified_.reset(nullptr);
  internal_event_notified_.reset(nullptr);
  waiting_.reset(nullptr);
  thread_queue_mapping_.clear();
  entities_.clear();
  int64_t total_time =  clock_.get()->timestamp() - start_timestamp_;
  (void)total_time;  // avoid unused variable warning
  GXF_LOG_INFO("Total execution time of EBS scheduler : %f ms\n", total_time/1000000.0);
  return thread_error_code_;
}

gxf_result_t EventBasedScheduler::prepare_abi(EntityExecutor* executor) {
  executor_ = executor;
  return GXF_SUCCESS;
}

void EventBasedScheduler::prepareResourceMapStrict(std::shared_ptr<ScheduleEntity> e) {
  Expected<Handle<ThreadPool>>
  entity_thread_pool = EntityResourceHelper::updateAndGetThreadPool(context(), e->eid_);
  // If effectively pinned entity, i.e. has thread pool associated AND the pool has a thread for it
  if (entity_thread_pool && entity_thread_pool.value()->getThread(e->eid_)) {
//    e->thread_pool_id_ = entity_thread_pool.?;
    e->thread_id_ = entity_thread_pool.value()->getThread(e->eid_)->uid;
    thread_pool_set_.emplace(entity_thread_pool.value().get());
  }
}

gxf_result_t EventBasedScheduler::schedule_abi(gxf_uid_t eid) {
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


  // Add entity name to lookup
  const char* entity_name = "UNKNOWN";
  GxfEntityGetName(context(), eid, &entity_name);

  std::shared_ptr<ScheduleEntity> e = std::make_shared<ScheduleEntity>(eid, entity_name);
  if (!e) { return GXF_FAILURE;}
  prepareResourceMapStrict(e);

  if (e->thread_id_ != kDefaultThreadPoolThreadId) {  // not default thread_pool
    if (thread_queue_mapping_.contains(e->thread_id_) == false) {
      // create a new queue for this thread
      ready_wait_time_jobs_.emplace_back(std::make_unique<TimedJobList<gxf_uid_t>>(
            [this]() -> int64_t { return clock_.get()->timestamp(); }));
      thread_queue_mapping_.insert(std::make_pair(e->thread_id_, ready_wait_time_jobs_.size() - 1));
    }
  }
  e->queue_index_ = thread_queue_mapping_[e->thread_id_].value();
  // Add scheduling condition to lookup
  const int64_t now = clock_.get()->timestamp();
  updateCondition(e, SchedulingCondition{SchedulingConditionType::READY, now});
  entities_[eid] = e;
  return GXF_SUCCESS;
}

gxf_result_t EventBasedScheduler::unschedule_abi(gxf_uid_t eid) {
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

  entities_[eid]->unschedule_ = true;
  return GXF_SUCCESS;
}

gxf_result_t EventBasedScheduler::runAsync_abi() {
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

  for (uint64_t i = 0; i < ready_wait_time_jobs_.size(); i++) {
    ready_wait_time_jobs_[i]->start();
  }


  // Creates the dispatcher thread
  dispatcher_thread_  = std::thread([this] { dispatcherThreadEntrance(); });

  // Creates async worker thread
  async_threads_.emplace_back([this] { asyncEventThreadEntrance(); });
  // Creates all worker threads
  for (const auto& thread_pool_ptr : thread_pool_set_) {
    for (const auto& thread_it : thread_pool_ptr->get()) {
      async_threads_.emplace_back([=] () { workerThreadEntrance(thread_pool_ptr,
                                                                thread_it.second.uid); });
      running_threads_.fetch_add(1);
    }
  }
  // Creates thread to stop after max_duration
  max_duration_thread_ = std::thread([=]() {
    auto max_duration = max_duration_ms_.try_get();
    if (max_duration) {
      std::unique_lock<std::mutex> lock(max_duration_sync_mutex_);
      GXF_LOG_INFO("Max duration thread started for %ld ms", max_duration.value());
      auto wait_val = max_duration_thread_cv_.wait_for(lock,
          std::chrono::milliseconds(*max_duration));
      if (wait_val ==  std::cv_status::timeout) {
        GXF_LOG_INFO("Max duration timeout %ld ms occurred", max_duration.value());
        stopAllJobs();
        std::unique_lock<std::mutex> lock(internal_event_notification_mutex_);
        GXF_LOG_DEBUG("Notifying internal event cv DUE TO TIMEOUT");
        internal_event_notification_cv_.notify_one();
        GXF_LOG_INFO("Event Based scheduler stopped.");
      }
    }
    return thread_error_code_;
  });
  return GXF_SUCCESS;
}

gxf_result_t EventBasedScheduler::stop_abi() {
  GXF_LOG_INFO("Stopping Event Based scheduler");
  stopAllJobs();
  // In order to join the dispatcher thread, it has to be in running state and not sleep state
  // notify dispatcher so that it can wake from its condition variable and return from its lambda
  // function
  notifyDispatcher();
  // dispatcher thread is responsible for cleaning up worker thread pool
  {
    // lock to avoid double join() call with wait_abi()
    std::unique_lock<std::mutex> lock(dispatcher_sync_mutex_);
    if (dispatcher_thread_.joinable()) { dispatcher_thread_.join(); }
  }
  // max_duration thread is responsible for exiting the scheduler when max duration expires
  {
    std::unique_lock<std::mutex> lock(max_duration_sync_mutex_);
    max_duration_thread_cv_.notify_one();
    if (max_duration_thread_.joinable()) { max_duration_thread_.join();}
  }
  GXF_LOG_INFO("Event Based scheduler stopped.");
  return thread_error_code_;
}

gxf_result_t EventBasedScheduler::wait_abi() {
  // sync work threads under lock since dispatcher can also sync them after execution is complete
  {
    std::unique_lock<std::mutex> lock(thread_sync_mutex_);
    thread_sync_cv_.wait(lock, [&] { return state_ != State::kRunning; });
    for (auto& thread : async_threads_) {
      if (thread.joinable()) { thread.join(); }
    }
  }

  {
    // lock to avoid double join() call with stop_abi()
    std::unique_lock<std::mutex> lock(dispatcher_sync_mutex_);
    if (dispatcher_thread_.joinable()) { dispatcher_thread_.join(); }
  }
  // max_duration thread is responsible for exiting the scheduler when max duration expires
  {
    std::unique_lock<std::mutex> lock(max_duration_sync_mutex_);
    max_duration_thread_cv_.notify_one();
    if (max_duration_thread_.joinable()) { max_duration_thread_.join();}
  }
  GXF_LOG_INFO("Event Based scheduler finished.");
  return thread_error_code_;
}

gxf_result_t EventBasedScheduler::event_notify_abi(gxf_uid_t eid, gxf_event_t event) {
  auto itr = entities_.find(eid);
  if (itr == entities_.end()) {
    return GXF_SUCCESS;
  }

  if (event == GXF_EVENT_EXTERNAL) {
    std::unique_lock<std::mutex> lock(external_event_notification_mutex_);
    external_event_notified_->pushEvent(eid);
    external_event_notification_cv_.notify_one();
  } else {
    notifyDispatcher(eid);
  }
  return GXF_SUCCESS;
}

// This function must not be called from any of the async threads since we join them here
gxf_result_t EventBasedScheduler::stopAllThreads() {
  GXF_LOG_INFO("Waiting to join all async threads");
  {
    std::unique_lock<std::mutex> lock(thread_sync_mutex_);
    for (auto& thread : async_threads_) {
      if (thread.joinable()) { thread.join(); }
    }
  }
  GXF_LOG_INFO("Waiting to join max duration thread");
  {
    std::unique_lock<std::mutex> lock(max_duration_sync_mutex_);
    max_duration_thread_cv_.notify_one();
  }
  if (max_duration_thread_.joinable()) { max_duration_thread_.join();}

  thread_sync_cv_.notify_all();
  GXF_LOG_INFO("All async worker threads joined, deactivating all entities");
  state_ = State::kStopped;
  return executor_->deactivateAll();  // Stop all the entities
}

void EventBasedScheduler::stopAllJobs() {
  GXF_LOG_INFO("Stopping all async jobs");
  state_ = State::kStopping;
  for (uint64_t i = 0; i < ready_wait_time_jobs_.size(); i++) {
    ready_wait_time_jobs_[i]->stop();                 // Stops worker threads
  }
  external_event_notified_->clear();             // Clear the event done list
  event_waiting_->clear();              // Clear the event waiting list
  {
    std::unique_lock<std::mutex> lock(external_event_notification_mutex_);
    external_event_notification_cv_.notify_one();  // Stops the async worker thread
  }
  return;
}

gxf_result_t EventBasedScheduler::stop_on_deadlock_timeout(const int64_t timeout,
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
