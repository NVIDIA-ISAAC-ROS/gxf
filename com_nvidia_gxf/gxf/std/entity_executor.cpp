/*
Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/entity_executor.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "common/nvtx_helper.hpp"
#include "gxf/std/scheduling_condition.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t EntityExecutor::initialize(Handle<Router> router,
                                        Handle<MessageRouter> message_router,
                                        Handle<NetworkRouter> network_router) {
  if (!router) { return GXF_ARGUMENT_NULL; }
  router_ = router;

  if (!message_router) { return GXF_ARGUMENT_NULL; }
  message_router_ = message_router;

  if (!network_router) { return GXF_ARGUMENT_NULL; }
  network_router_ = network_router;

  statistics_ = std::make_shared<FixedVector<Handle<JobStatistics>, kMaxComponents>>
    (FixedVector<Handle<JobStatistics>, kMaxComponents>());

  nvtx_domain_ = nvtxDomainCreate("GXF");

  return GXF_SUCCESS;
}

gxf_result_t EntityExecutor::activate(gxf_context_t context, gxf_uid_t eid) {
  Expected<Entity> entity = Entity::Shared(context, eid);
  if (!entity) {
    return entity.error();
  }

  // Create a new entity and try to activate it.
  auto item = std::make_unique<EntityItem>();
  const Expected<bool> activatable =
      item->activate(std::move(entity.value()), message_router_, statistics_,
        nvtx_domain_, ++nvtx_category_id_last_);
  if (!activatable) {
    return activatable.error();
  }

  // The item does not have to be activated.
  if (!activatable.value()) {
    return GXF_SUCCESS;
  }

  // Added to active items
  std::unique_lock<std::mutex> lock(mutex_);
  items_.emplace(eid, std::move(item));

  return GXF_SUCCESS;
}

gxf_result_t EntityExecutor::deactivate(gxf_uid_t eid) {
  // Find and remove the item under lock
  std::unique_ptr<EntityItem> item;
  {
    std::unique_lock<std::mutex> lock(mutex_);

    const auto it = items_.find(eid);
    if (it == items_.end()) {
      // Not an error as non-activatable entities are not tracked. Deactivating twice is thus
      // ignored.
      return GXF_SUCCESS;
    }

    item = std::move(it->second);
    items_.erase(it);
  }

  // Deactivate the item outside of the lock. This call is stopping codelets and can potentially
  // lead to recusrive deactivations.
  item->deactivate();

  return GXF_SUCCESS;
}

gxf_result_t EntityExecutor::deactivateAll() {
  // Get a copy of the items under lock and clear the list.
  std::map<gxf_uid_t, std::unique_ptr<EntityItem>> items;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    items = std::move(items_);
    items_.clear();
  }

  // Deactivate the item outside of the lock. This call is stopping codelets and can potentially
  // lead to recusrive deactivations.
  Expected<void> code;
  for (auto& kvp : items) {
    code = AccumulateError(code, kvp.second->deactivate());
  }

  return ToResultCode(code);
}

Expected<void> EntityExecutor::getEntities(FixedVectorBase<gxf_uid_t>& entities) const {
  std::unique_lock<std::mutex> lock(mutex_);

  entities.clear();
  for (const auto& kvp : items_) {
    auto result = entities.push_back(kvp.first);
    if (!result) {
      GXF_LOG_WARNING("Exceeding container capacity");
      return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE};
    }
  }

  return Success;
}

gxf_result_t EntityExecutor::setClock(Handle<Clock> clock) {
  if (!clock) { return GXF_ARGUMENT_NULL; }
  Expected<void> result = Success;
  result &= router_->setClock(clock);
  result &= message_router_->setClock(clock);
  if (!result) return ToResultCode(result);
  return GXF_SUCCESS;
}

Expected<SchedulingCondition> EntityExecutor::executeEntity(gxf_uid_t eid, int64_t timestamp) {
  EntityItem* item;
  {
    std::unique_lock<std::mutex> lock(mutex_);

    const auto it = items_.find(eid);
    if (it == items_.end()) {
      return Unexpected{GXF_ENTITY_NOT_FOUND};
    }
    item = it->second.get();
  }

  // prepare statistics collection
  const bool entity_started = (item->getEntityStatus().value() == GXF_ENTITY_STATUS_STARTED) ||
                              (item->getEntityStatus().value() == GXF_ENTITY_STATUS_IDLE);
  if (entity_started) {
    std::lock_guard<std::mutex> lock(statistics_mutex_);
    for (size_t i = 0; i < statistics_->size(); i++) {
      statistics_->at(i).value()->preJob(eid);
    }
  }

  // Difference between the actual execution timestamp and the target execution timestamp
  int64_t ticking_variation = 0;

  // execute the entity
  auto result = item->execute(timestamp, router_, ticking_variation);

  // iterate through entity monitors
  {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    for (size_t i = 0; i < monitors_.size(); i++) {
      monitors_.at(i).value()->onExecute(item->entity.eid(), timestamp, ToResultCode(result));
    }
  }

  // finalize statistics collection
  if (entity_started && result && result.value().type == SchedulingConditionType::READY) {
    std::lock_guard<std::mutex> lock(statistics_mutex_);
    for (size_t i = 0; i < statistics_->size(); i++) {
      statistics_->at(i).value()->postJob(eid, ticking_variation);
    }
  }
  return result;
}

Expected<SchedulingCondition> EntityExecutor::checkEntity(gxf_uid_t eid, int64_t timestamp) {
  EntityItem* item;
  {
    std::unique_lock<std::mutex> lock(mutex_);

    const auto it = items_.find(eid);
    if (it == items_.end()) {
      return Unexpected{GXF_ENTITY_NOT_FOUND};
    }
    item = it->second.get();
  }
  return item->check(timestamp);
}

gxf_result_t EntityExecutor::getEntityStatus(gxf_uid_t eid, gxf_entity_status_t* entity_status) {
  EntityItem* item;
  {
    std::unique_lock<std::mutex> lock(mutex_);

    const auto it = items_.find(eid);
    if (it == items_.end()) {
      GXF_LOG_ERROR("Entity with eid %ld not found!", eid);
      return GXF_ENTITY_NOT_FOUND;
    }
    item = it->second.get();
  }
  auto status = item->getEntityStatus();
  if (!status) { return status.error(); }
  *entity_status = status.value();
  return GXF_SUCCESS;
}

gxf_result_t EntityExecutor::getEntityBehaviorStatus(gxf_uid_t eid,
                                                     entity_state_t& behavior_status) {
  EntityItem* item;
  {
    std::unique_lock<std::mutex> lock(mutex_);

    const auto it = items_.find(eid);
    if (it == items_.end()) {
      GXF_LOG_ERROR("Entity with eid %ld not found!", eid);
      return GXF_ENTITY_NOT_FOUND;
    }
    item = it->second.get();
  }
  behavior_status = item->behavior_status;
  return GXF_SUCCESS;
}

Expected<SchedulingCondition> EntityExecutor::EntityItem::check(int64_t timestamp) const {
  SchedulingCondition combined{SchedulingConditionType::READY, 0};
  for (size_t i = 0; i < terms.size(); i++) {
    auto& term = terms.at(i).value();
    Expected<SchedulingCondition> result = term->check(timestamp);
    if (!result) { return ForwardError(result); }
    for (size_t i = 0; i < statistics_->size(); ++i) {
      statistics_->at(i).value()->postTermCheck(entity.eid(), term.cid(),
                                                SchedulingConditionTypeStr(result.value().type));
    }
    combined = AndCombine(combined, result.value());
  }
  return combined;
}

Expected<bool> EntityExecutor::EntityItem::activate(
    Entity other, MessageRouter* message_router,
    std::shared_ptr<FixedVector<Handle<JobStatistics>, kMaxComponents>> statistics,
    nvtxDomainHandle_t nvtx_domain, uint32_t nvtx_category_id) {
  if (message_router == nullptr) { return Unexpected{GXF_ARGUMENT_NULL}; }

  entity = std::move(other);
  this->statistics_ = statistics;

  nvtx_domain_ = nvtx_domain;
  nvtx_category_id_ = nvtx_category_id;

  std::string nvtx_category_name = "Entity: " + std::string(entity.name());
  nvtxDomainNameCategory(nvtx_domain_, nvtx_category_id_, nvtx_category_name.c_str());

  setEntityStatus(GXF_ENTITY_STATUS_NOT_STARTED);

  auto result = entity.findAll<PeriodicSchedulingTerm>()
      .assign_to(periodic_terms)
      .and_then([&]() { return entity.findAll<DownstreamReceptiveSchedulingTerm>(); })
      .assign_to(downstream_receptive_terms)
      .and_then([&]() { return entity.findAll<SchedulingTerm>(); })
      .assign_to(terms)
      .and_then([&]() { return entity.findAll<Codelet>(); })
      .assign_to(codelets);
  if (!result) {
    return ForwardError(result);
  }

  for (size_t i = 0; i < downstream_receptive_terms.size(); i++) {
    const auto& term = downstream_receptive_terms.at(i).value();
    // Find the receiver which is connected to the transmitter and give it to the scheduling
    // term.
    if (const auto rx = message_router->getRx(term->transmitter())) {
      term->setReceiver(rx.value());
    } else {
      // Not connected => term will never succeed
      GXF_LOG_ERROR(
          "[E%05zu] No receiver connected to transmitter of DownstreamReceptiveSchedulingTerm "
          "%zu of entity \"%s\". The entity will never tick.",
          entity.eid(), term.cid(), entity.name());
    }
  }

  return !codelets.empty();
}

Expected<SchedulingCondition> EntityExecutor::EntityItem::execute(
    int64_t timestamp, Router* router, int64_t& ticking_variation) {
  if (router == nullptr) {
    return Unexpected{GXF_ARGUMENT_NULL};
  }

  // The entity must be in a valid state
  if (status_ == GXF_ENTITY_STATUS_START_PENDING) {
    GXF_LOG_ERROR("Entity %s cannot be executed before being started", entity.name());
    return Unexpected{GXF_INVALID_EXECUTION_SEQUENCE};
  }
  if (status_ == GXF_ENTITY_STATUS_TICK_PENDING) {
    GXF_LOG_ERROR("Entity %s is already waiting to be executed", entity.name());
    return Unexpected{GXF_INVALID_EXECUTION_SEQUENCE};
  }
  if (status_ == GXF_ENTITY_STATUS_STOP_PENDING) {
    GXF_LOG_ERROR("Entity %s cannot be executed since it is being stopped", entity.name());
    return Unexpected{GXF_INVALID_EXECUTION_SEQUENCE};
  }

  std::unique_lock<std::mutex> lock(mutex_);

  // If the entity was not started yet, then start it now.
  if (status_ == GXF_ENTITY_STATUS_NOT_STARTED) {
    return start(timestamp).substitute(SchedulingCondition{SchedulingConditionType::READY,
                                                           timestamp});
  }

  // Check the current scheduling status of the entity.
  const auto maybe_condition = check(timestamp);
  if (!maybe_condition) {
    stop();  // FIXME Should we do anything with the error?
    return ForwardError(maybe_condition);
  }
  SchedulingCondition condition = maybe_condition.value();

  // In case of a timed wait event we consider last execution timestamp to determine if the
  // entity is ready to tick.
  if (condition.type == SchedulingConditionType::WAIT_TIME) {
    const int64_t target = condition.target_timestamp;
    if (target <= timestamp) {
      condition = {SchedulingConditionType::READY, target};
    } else {
      return SchedulingCondition{SchedulingConditionType::WAIT_TIME, target};
    }
  }

  if (condition.type == SchedulingConditionType::READY) {
    ticking_variation = timestamp - condition.target_timestamp;
    // ready to tick
    setEntityStatus(GXF_ENTITY_STATUS_TICK_PENDING);  // FIXME(v1) set in a thread-safe manner

    const auto code = tick(timestamp, router);

    // find the controller
    gxf_tid_t tid;
    gxf_uid_t cid;

    if (controller.is_null()) {
      // Try to find the controller first
      GXF_ASSERT_SUCCESS(GxfComponentTypeId(entity.context(),
                                            "nvidia::gxf::Controller", &tid));
      const auto componentFindResult = GxfComponentFind(
          entity.context(), entity.eid(), tid, nullptr, nullptr, &cid);
      if (componentFindResult == GXF_SUCCESS) {
        auto result = Handle<Controller>::Create(entity.context(), cid);
        if (!result) {
          return Unexpected{GXF_FAILURE};
        }
        controller = result.value();
      }
    }

    if (!controller.is_null()) {
      // if there is a controller, use controller's termination policy
      gxf_controller_status_t controller_status;

      controller_status = controller->control(entity.eid(), code);

      // store behavior status inside EntityItem
      behavior_status = controller_status.behavior_status;

      // terminate according to the returned execution status
      switch (controller_status.exec_status) {
        case GXF_EXECUTE_SUCCESS:
          if (behavior_status == GXF_BEHAVIOR_RUNNING) {
            setEntityStatus(GXF_ENTITY_STATUS_STARTED);
            return SchedulingCondition{SchedulingConditionType::READY, timestamp};
          } else {
            return SchedulingCondition{SchedulingConditionType::NEVER, 0};
          }
        case GXF_EXECUTE_FAILURE_DEACTIVATE:
          // GXF_BEHAVIOR_FAILURE => deactivate
          setEntityStatus(GXF_ENTITY_STATUS_STARTED);
          stop();
          GXF_LOG_INFO(
              "Deactivating the entity after failures. Behavior status is: %d",
              behavior_status);
          return SchedulingCondition{SchedulingConditionType::NEVER, 0};
        case GXF_EXECUTE_FAILURE_REPEAT:
          // GXF_BEHAVIOR_FAILURE => will repeat
          setEntityStatus(GXF_ENTITY_STATUS_STARTED);
          GXF_LOG_INFO("Repeating after failure");
          return SchedulingCondition{SchedulingConditionType::READY, timestamp};
        case GXF_EXECUTE_FAILURE:
          // deactivate all the entities = stop the entire graph execution
          // only return GXF_FAILURE to scheduler in this case
          return Unexpected{GXF_FAILURE};
        default:
          return Unexpected{GXF_FAILURE};
      }
    }

    // if there's no controller use default termination policy
    if (!code) {
      stop();
      return ForwardError(code);
    } else {
      return condition;
    }
  } else if (condition.type == SchedulingConditionType::WAIT ||
             condition.type == SchedulingConditionType::WAIT_EVENT) {
    // not ready to step
    return condition;
  } else if (condition.type == SchedulingConditionType::NEVER) {
    // won't ever step again
    return stop().substitute(condition);
  } else {
    return Unexpected{GXF_INVALID_ENUM};
  }
}

Expected<void> EntityExecutor::EntityItem::deactivate() {
  std::unique_lock<std::mutex> lock(mutex_);
  // As this is executed under lock we are not in a state where an operation is pending.

  // If we the entity was not started there is nothing to do.
  if (status_ == GXF_ENTITY_STATUS_NOT_STARTED) {
    return Success;
  }

  GXF_LOG_VERBOSE("Deactivating entity name:[%s] eid:[%lu]]", entity.name(), entity.eid());
  auto result = stop();
  nvtxDomainDestroy(nvtx_domain_);
  return result;
}

Expected<void> EntityExecutor::EntityItem::start(int64_t timestamp) {
  if (status_ != GXF_ENTITY_STATUS_NOT_STARTED) {
    GXF_LOG_ERROR("Entity must be in GXF_ENTITY_STATUS_NOT_STARTED stage before starting."
                  " Current state is %s", entityStatusStr(status_));
    return Unexpected{GXF_INVALID_LIFECYCLE_STAGE};
  }

  setEntityStatus(GXF_ENTITY_STATUS_START_PENDING);

  // initialize execution times
  for (size_t i = 0; i < codelets.size(); i++) {
    codelets.at(i).value()->beforeStart(timestamp);
  }

  for (size_t i = 0; i < codelets.size(); i++) {
    std::string nvtx_string = "codelet start: " + std::string(codelets.at(i)->name());
    auto red_marker = CreateRedEvent(nvtx_string, nvtx_category_id_);
    // Theoretically we can also use `nvtxDomainRangePush/Pop` here since the range is definitely
    // started and ended on the current thread. However since the activity of an entity can happen
    // across different threads, we use `nvtxDomainRangeStart/End` here to make every range visible
    // on the process-wide NVTX row.
    nvtx_range_codelet_start_ = nvtxDomainRangeStartEx(nvtx_domain_, &red_marker);
    const Expected<void> code = startCodelet(codelets.at(i).value());
    nvtxDomainRangeEnd(nvtx_domain_, nvtx_range_codelet_start_);
    if (!code) {
      // FIXME (v1) stop codelets which started successfully so far.
      return code;
    }
  }

  setEntityStatus(GXF_ENTITY_STATUS_STARTED);
  last_execution_timestamp_ = timestamp;

  return Success;
}

Expected<void> EntityExecutor::EntityItem::tick(int64_t timestamp, Router* router) {
  Expected<void> code;

  if (status_ != GXF_ENTITY_STATUS_TICK_PENDING) {
    GXF_LOG_ERROR("Entity [%s] must be in GXF_ENTITY_STATUS_TICK_PENDING stage before ticking."
                  " Current state is %s", entity.name(), entityStatusStr(status_));
    return Unexpected{GXF_INVALID_LIFECYCLE_STAGE};
  }

  if (timestamp < last_execution_timestamp_) {
    GXF_LOG_ERROR("Entity [%s] - Received execution target timestamp %ld lesser than last "
                  "execution timestamp %ld", entity.name(), timestamp, last_execution_timestamp_);
    return Unexpected{GXF_FAILURE};
  }

  // flush the input queues
  code = router->syncInbox(entity);
  if (!code) {
    GXF_LOG_ERROR("Failed to sync inbox for entity: %s code: %s", entity.name(),
                   GxfResultStr(code.error()));
    return code;
  }

  // update execution times
  for (size_t i = 0; i < codelets.size(); i++) {
    codelets.at(i).value()->beforeTick(timestamp);
  }

  setEntityStatus(GXF_ENTITY_STATUS_TICKING);
  // execute codelets
  for (size_t i = 0; i < codelets.size(); i++) {
    // try {  // FIXME Enable exceptions only via define.
    std::string nvtx_string = "tick codelet: " + std::string(codelets.at(i)->name());
    auto green_marker = CreateGreenEvent(nvtx_string, nvtx_category_id_);
    // Theoretically we can also use `nvtxDomainRangePush/Pop` here since the range is definitely
    // started and ended on the current thread. However since the activity of an entity can happen
    // across different threads, we use `nvtxDomainRangeStart/End` here to make every range visible
    // on the process-wide NVTX row.
    nvtx_range_tick_codelet_ = nvtxDomainRangeStartEx(nvtx_domain_, &green_marker);
    code = tickCodelet(codelets.at(i).value());
    nvtxDomainRangeEnd(nvtx_domain_, nvtx_range_tick_codelet_);
    // } catch (const std::runtime_error& e) {
    //   GXF_LOG_ERROR("Failed to step entity: %s", e.what());
    //   return Unexpected{GXF_FAILURE};
    // }
    if (!code) {
      GXF_LOG_ERROR("Failed to tick codelet %s in entity: %s code: %s",
                     codelets.at(i).value()->name(), entity.name(), GxfResultStr(code.error()));
      return ForwardError(code);
    }
  }

  // update scheduling terms
  for (size_t i = 0; i < terms.size(); i++) {
    code = terms.at(i).value()->onExecute(timestamp);
    if (!code) {
      return code;
    }
  }

  // flush the output queues and distribute messages
  code = router->syncOutbox(entity);
  if (!code) {
    GXF_LOG_ERROR("Failed to sync outbox for entity: %s code: %s", entity.name(),
                  GxfResultStr(code.error()));
    return code;
  }

  // mark time of execution and status to idle
  setEntityStatus(GXF_ENTITY_STATUS_IDLE);
  last_execution_timestamp_ = timestamp;

  return Success;
}

Expected<void> EntityExecutor::EntityItem::stop() {
  if ((status_ != GXF_ENTITY_STATUS_STARTED) &&
      (status_ != GXF_ENTITY_STATUS_IDLE) &&
      (status_ != GXF_ENTITY_STATUS_TICK_PENDING) &&
      (status_ != GXF_ENTITY_STATUS_TICKING)) {
    GXF_LOG_ERROR("Entity [%s] must be in Started, Tick Pending, Ticking or Idle"
                  " stage before stopping. Current state is %s",
                  entity.name(), entityStatusStr(status_));
    return Unexpected{GXF_INVALID_LIFECYCLE_STAGE};
  }

  setEntityStatus(GXF_ENTITY_STATUS_STOP_PENDING);

  // update execution times
  for (size_t i = 0; i < codelets.size(); i++) {
    codelets.at(i).value()->beforeStop();
  }

  Expected<void> code = Success;
  for (size_t i = 0; i < codelets.size(); i++) {
    std::string nvtx_string = "codelet stop: " + std::string(codelets.at(i)->name());
    auto black_marker = CreateBlackEvent(nvtx_string, nvtx_category_id_);
    // Theoretically we can also use `nvtxDomainRangePush/Pop` here since the range is definitely
    // started and ended on the current thread. However since the activity of an entity can happen
    // across different threads, we use `nvtxDomainRangeStart/End` here to make every range visible
    // on the process-wide NVTX row.
    nvtx_range_codelet_stop_ = nvtxDomainRangeStartEx(nvtx_domain_, &black_marker);
    code &= stopCodelet(codelets.at(i).value());
    nvtxDomainRangeEnd(nvtx_domain_, nvtx_range_codelet_stop_);
  }

  setEntityStatus(GXF_ENTITY_STATUS_NOT_STARTED);

  return code;
}

Expected<void> EntityExecutor::addStatistics(Handle<JobStatistics> statistics) {
  std::lock_guard<std::mutex> lock(statistics_mutex_);

  auto result = statistics_->push_back(statistics);
  if (!result) {
    GXF_LOG_WARNING("Exceeding maximum number of JobStatistics");
    return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE};
  }
  return Success;
}

Expected<void> EntityExecutor::removeStatistics(Handle<JobStatistics> statistics) {
  std::lock_guard<std::mutex> lock(statistics_mutex_);
  for (size_t i = 0; i < statistics_->size(); ++i) {
    if (statistics == statistics_->at(i).value()) {
      statistics_->erase(i);
      return Success;
    }
  }
  return Unexpected{GXF_ENTITY_COMPONENT_NOT_FOUND};
}

Expected<void> EntityExecutor::addMonitor(Handle<Monitor> monitor) {
  std::lock_guard<std::mutex> lock(monitor_mutex_);
  auto result = monitors_.push_back(monitor);
  if (!result) {
    GXF_LOG_WARNING("Exceeding maximum number of Monitors");
    return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE};
  }
  return Success;
}

Expected<void> EntityExecutor::removeMonitor(Handle<Monitor> monitor) {
  std::lock_guard<std::mutex> lock(monitor_mutex_);
  for (size_t i = 0; i < monitors_.size(); ++i) {
    if (monitor == monitors_.at(i).value()) {
      monitors_.erase(i);
      return Success;
    }
  }
  return Unexpected{GXF_ENTITY_COMPONENT_NOT_FOUND};
}

Expected<void> EntityExecutor::EntityItem::startCodelet(const Handle<Codelet>& codelet) {
  GXF_LOG_DEBUG("[C%05zu] starting codelet '%s/%s'", codelet->cid(), codelet->entity().name(),
               codelet->name());
  return ExpectedOrCode(codelet->start());
}

Expected<void> EntityExecutor::EntityItem::tickCodelet(const Handle<Codelet>& codelet) {
  GXF_LOG_DEBUG("[C%05zu] tick codelet %s", codelet->cid(), codelet->name());

  // Lets check if statistics_ is valid and contains statistics components.
  // Collect codelet statistics if requested
  if (statistics_ && statistics_->size() != 0) {
      for (size_t i = 0; i < statistics_->size(); i++) {
        if (statistics_->at(i).value()->isCodeletStatistics()) {
          statistics_->at(i).value()->preTick(codelet->eid(), codelet->cid());
        }
    }

    auto code = codelet->tick();
    if (GXF_SUCCESS != code) { return ExpectedOrCode(code); }

    for (size_t i = 0; i < statistics_->size(); i++) {
      if (statistics_->at(i).value()->isCodeletStatistics()) {
        statistics_->at(i).value()->postTick(codelet->eid(), codelet->cid());
      }
    }
    return Success;
  }

  return ExpectedOrCode(codelet->tick());
}

Expected<void> EntityExecutor::EntityItem::stopCodelet(const Handle<Codelet>& codelet) {
  GXF_LOG_DEBUG("[C%05zu] stop codelet %s", codelet->cid(), codelet->name());
  return ExpectedOrCode(codelet->stop());
}

Expected<gxf_entity_status_t> EntityExecutor::EntityItem::getEntityStatus() {
  return status_;
}

Expected<void> EntityExecutor::EntityItem::setEntityStatus(const gxf_entity_status_t next_state) {
  if (!first_status_change_) {
    if (status_ != GXF_ENTITY_STATUS_NOT_STARTED && status_ != GXF_ENTITY_STATUS_IDLE) {
      nvtxDomainRangeEnd(nvtx_domain_, nvtx_range_entity_state_);
    }
  } else {
    first_status_change_ = false;
  }

  status_ = next_state;
  Expected<void> result;
  for (size_t i = 0; i < statistics_->size(); ++i) {
      statistics_->at(i).value()->onLifecycleChange(entity.eid(), entityStatusStr(next_state));
  }

  // We do not emit NVTX ranges for "NotStarted" and "Idle" states because they will occupy
  // too much space on profiling tools' timeline, making it hard to find more useful activties.
  if (next_state != GXF_ENTITY_STATUS_NOT_STARTED && next_state != GXF_ENTITY_STATUS_IDLE) {
    std::string nvtx_string = "entity state update: " + std::string(entityStatusStr(next_state));
    auto blue_marker = CreateBlueEvent(nvtx_string, nvtx_category_id_);
    // It is possible for a state to start in one thread, but end in a different thread.  Therefore
    // we cannot use `nvtxDomainRangePush/Pop` here and must use `nvtxDomainRangeStart/End` to emit
    // process-wide ranges.
    nvtx_range_entity_state_ = nvtxDomainRangeStartEx(nvtx_domain_, &blue_marker);
  }

  return Success;
}

const char* EntityExecutor::EntityItem::entityStatusStr(gxf_entity_status_t status) {
  switch (status) {
    GXF_ENUM_TO_STR(GXF_ENTITY_STATUS_NOT_STARTED, NotStarted)
    GXF_ENUM_TO_STR(GXF_ENTITY_STATUS_START_PENDING, StartPending)
    GXF_ENUM_TO_STR(GXF_ENTITY_STATUS_STARTED, Started)
    GXF_ENUM_TO_STR(GXF_ENTITY_STATUS_TICK_PENDING, Pending)
    GXF_ENUM_TO_STR(GXF_ENTITY_STATUS_TICKING, Ticking)
    GXF_ENUM_TO_STR(GXF_ENTITY_STATUS_IDLE, Idle)
    GXF_ENUM_TO_STR(GXF_ENTITY_STATUS_STOP_PENDING, StopPending)
    default:
      return "N/A";
  }
}

}  // namespace gxf
}  // namespace nvidia
