/*
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/program.hpp"

#include <memory>
#include <shared_mutex>  // NOLINT
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/type_name.hpp"
#include "gxf/core/gxf.h"
#include "gxf/std/connection.hpp"
#include "gxf/std/job_statistics.hpp"
#include "gxf/std/message_router.hpp"
#include "gxf/std/monitor.hpp"
#include "gxf/std/network_context.hpp"
#include "gxf/std/resources.hpp"
#include "gxf/std/router_group.hpp"
#include "gxf/std/scheduler.hpp"
#include "gxf/std/system.hpp"

namespace nvidia {
namespace gxf {

Program::Program() {
  state_ = State::ORIGIN;
  context_ = nullptr;
  entity_warden_ = nullptr;
  entity_executor_ = nullptr;
}

Expected<void> Program::setup(gxf_context_t context, EntityWarden* warden,
                              EntityExecutor* executor,
                              std::shared_ptr<ParameterStorage> parameter_storage) {
  if (context == nullptr) { return Unexpected{GXF_ARGUMENT_NULL}; }
  if (warden == nullptr) { return Unexpected{GXF_ARGUMENT_NULL}; }
  if (executor == nullptr) { return Unexpected{GXF_ARGUMENT_NULL}; }

  context_ = context;
  entity_warden_ = warden;
  entity_executor_ = executor;
  parameter_storage_ = parameter_storage;

  unscheduled_entities_.reserve(kMaxEntities);
  scheduled_entities_.reserve(kMaxEntities);

  return Success;
}

Expected<void> Program::addEntity(gxf_uid_t eid, EntityItem* item_ptr) {
  std::lock_guard<std::recursive_mutex> lock(entity_mutex_);
  auto maybe = Entity::Shared(context_, eid, item_ptr);
  if (!maybe) { return ForwardError(maybe); }

  // Entity is unscheduled when it's created
  unscheduled_entities_.emplace_back(std::move(maybe.value()));
  return Success;
}

Expected<void> Program::activate() {
  // Prevent new entities from getting added until graph is activated
  std::lock_guard<std::recursive_mutex> lock(entity_mutex_);

  // Make sure that necessary extensions were loaded.
  const gxf_result_t code = GxfComponentTypeId(context_, TypenameAsString<System>(), &sys_tid_);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("'System' component not registered. Did you load the STD extensions?");
    return Unexpected{code};
  }

  State origin = State::ORIGIN;
  if (!state_.compare_exchange_strong(origin, State::ACTIVATING)) {
    GXF_LOG_ERROR("Unexpected State: %hhd", static_cast<int32_t>(state_.load()));
    return Unexpected{GXF_INVALID_EXECUTION_SEQUENCE};
  }

  // Sort entities based on their components so they can
  // be activated in order
  FixedVector<Entity> graph_entities;
  FixedVector<Entity> connection_entities;
  FixedVector<Entity> system_entities;
  FixedVector<Entity> router_entities;
  FixedVector<Entity> network_context_entities;

  graph_entities.reserve(kMaxEntities);
  connection_entities.reserve(kMaxEntities);
  system_entities.reserve(kMaxEntities);
  router_entities.reserve(kMaxEntities);
  network_context_entities.reserve(kMaxEntities);

  for (size_t i = 0; i < unscheduled_entities_.size(); ++i) {
    const gxf_uid_t eid = unscheduled_entities_.at(i).value().eid();
    Expected<Entity> maybe = Entity::Shared(context_, eid);
    if (!maybe) {
      const char* name = "UNKNOWN";
      GxfEntityGetName(context_, eid, &name);
      GXF_LOG_ERROR("Failed to create shared entity from unscheduled entity "
                    "with eid %05zu named %s", eid, name);
      return ForwardError(maybe);
    }
    auto entity = maybe.value();
    auto systems = entity.findAllHeap<System>();
    if (!systems) {
      return ForwardError(systems);
    }
    auto connections = entity.findAllHeap<Connection>();
    if (!connections) {
      return ForwardError(connections);
    }
    auto routers = entity.findAllHeap<Router>();
    if (!routers) {
      return ForwardError(routers);
    }
    auto network_contexts = entity.findAllHeap<NetworkContext>();
    if (!network_contexts) {
      return ForwardError(network_contexts);
    }
    if (!systems->empty()) {
      auto result = system_entities.emplace_back(entity);
      if (!result) { return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE}; }
    } else if (!routers->empty()) {
      auto result = router_entities.emplace_back(entity);
      if (!result) { return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE}; }
    } else if (!connections->empty()) {
      auto result = connection_entities.emplace_back(entity);
      if (!result) { return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE}; }
    } else if (!network_contexts->empty()) {
      auto result = network_context_entities.emplace_back(entity);
      if (!result) { return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE}; }
    } else {
      auto result = graph_entities.emplace_back(entity);
      if (!result) { return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE}; }
    }
  }

  // Create system group entity
  {
    auto maybe = Entity::New(context_);
    if (!maybe) {
      GXF_LOG_ERROR("Failed to create system group entity");
      return ForwardError(maybe);
    }
    system_group_entity_ = std::move(maybe.value());
  }
  {
    auto maybe = system_group_entity_.add<SystemGroup>("system_group");
    if (!maybe) {
      GXF_LOG_ERROR("Failed to add system group entity");
      return ForwardError(maybe);
    }
    system_group_ = std::move(maybe.value());
  }

  // Create router group entity
  Handle<MessageRouter> message_router;
  Handle<NetworkRouter> network_router;
  {
    auto maybe = Entity::New(context_);
    if (!maybe) {
      GXF_LOG_ERROR("Failed to create router group entity");
      return ForwardError(maybe);
    }
    router_group_entity_ = std::move(maybe.value());
  }
  {
    auto maybe = router_group_entity_.add<RouterGroup>("router_group");
    if (!maybe) {
      GXF_LOG_ERROR("Failed to add router group to router group entity");
      return ForwardError(maybe);
    }
    router_group_ = std::move(maybe.value());
  }
  {
    auto maybe = router_group_entity_.add<MessageRouter>("message_router");
    if (!maybe) {
      GXF_LOG_ERROR("Failed to add message router to router group entity");
      return ForwardError(maybe);
    }
    message_router = std::move(maybe.value());
  }
  auto result = router_group_->addRouter(message_router);
  if (!result) {
    GXF_LOG_ERROR("Failed to add message router to the router group");
     return ForwardError(result);
  }
  {
    auto maybe = router_group_entity_.add<NetworkRouter>("network_router");
    if (!maybe) {
      GXF_LOG_ERROR("Failed to add network router to router group entity");
      return ForwardError(maybe);
    }
    network_router = std::move(maybe.value());
  }
  result = router_group_->addRouter(network_router);
  if (!result) {
    GXF_LOG_ERROR("Failed to add network router to the router group");
     return ForwardError(result);
  }

  for (const auto& entity : router_entities) {
    // Find all routers
    auto routers = entity->findAllHeap<Router>();
    if (!routers) { return ForwardError(routers); }
    for (auto router : routers.value()) {
      if (!router) {
        GXF_LOG_ERROR("Found a bad router component while scheduling entity %s", entity->name());
        return Unexpected{GXF_FAILURE};
      }
      auto result = router_group_->addRouter(router.value());
      if (!result) { return ForwardError(result); }
    }
  }
  // Add network context to router
  for (size_t i = 0; i < unscheduled_entities_.size(); i++) {
    const gxf_uid_t eid = unscheduled_entities_.at(i)->eid();
    const auto maybe_entity = Entity::Shared(context_, eid);
    if (!maybe_entity) { ForwardError(maybe_entity); }
    auto network_contexts = maybe_entity->findAllHeap<NetworkContext>();
    for (auto context : network_contexts.value()) {
      if (context) {
        router_group_->addNetworkContext(context.value());
      }
    }
  }

  // add message routes from all entities
  const auto add_routes_result = addRoutes(unscheduled_entities_);
  if (!add_routes_result) { return add_routes_result; }

  // initialize the entity executor with the routers
  entity_executor_->initialize(router_group_, message_router, network_router);

  // pre activate 5. graph_entities only
  // assumption is no resource components placed in entities of
  // 1. scheduler, 2. routers, 3. connections, 4. network
  auto pre_activation_result = preActivateEntities(graph_entities);
  if (!pre_activation_result) { return pre_activation_result; }

  // Order of entity activation
  // 1. schedulers - must be init first to receive scheduling requests for other entities
  // 2. routers - must be init before connections
  // 3. connections - setup message routes with the router group
  // 4. network contexts (Ucx Context)
  // 5. graph entities (codelets and other components)
  auto activation_result = activateEntities(std::move(system_entities))
                  .and_then([&](){ return activateEntities(std::move(router_entities)); })
                  .and_then([&](){ return activateEntities(std::move(connection_entities)); })
                  .and_then([&](){ return activateEntities(std::move(network_context_entities)); })
                  .and_then([&](){ return activateEntities(std::move(graph_entities)); });
  if (!activation_result) { return activation_result; }

  state_ = State::ACTIVATED;
  return Success;
}

Expected<void> Program::addRoutes(const FixedVector<Entity>& entities) {
  for (size_t i = 0; i < entities.size(); i++) {
    const gxf_uid_t eid = entities.at(i)->eid();
    const auto maybe_entity = Entity::Shared(context_, eid);
    if (!maybe_entity) { ForwardError(maybe_entity); }
    const auto add_route_result = router_group_->addRoutes(maybe_entity.value());
    if (!add_route_result) {
      GXF_LOG_ERROR("Failed to activate entity %s.", maybe_entity.value().name());
      GXF_LOG_ERROR("Deactivating...");
      const auto deactivate_result = deactivate();
      if (!deactivate_result) { GXF_LOG_ERROR("Deactivation failed."); }
      return deactivate_result;
    }
  }
  return Success;
}

Expected<void> Program::activateEntities(FixedVector<Entity> entities) {
  for (size_t i = 0; i < entities.size(); i++) {
    const gxf_uid_t eid = entities.at(i)->eid();
    const gxf_result_t code = GxfEntityActivate(context_, eid);
    if (code != GXF_SUCCESS) {
      const char* entityName = "UNKNOWN";
      GxfEntityGetName(context_, eid, &entityName);
      GXF_LOG_ERROR("Failed to activate entity %05zu named %s: %s",
        eid, entityName, GxfResultStr(code));
      GXF_LOG_ERROR("Deactivating...");
      auto result = deactivate();
      if (!result) { GXF_LOG_ERROR("Deactivation failed."); }
      return Unexpected{code};
    }
  }

  return Success;
}

Expected<void> Program::preActivateEntities(const FixedVector<Entity>& entities) {
  for (size_t i = 0; i < entities.size(); i++) {
    const gxf_uid_t eid = entities.at(i)->eid();
    const gxf_result_t code = entity_warden_->populateResourcesToEntityGroup(context_, eid);
    if (code != GXF_SUCCESS) {
      const char* entityName = "UNKNOWN";
      GxfEntityGetName(context_, eid, &entityName);
      GXF_LOG_ERROR(
        "Failed to populate resources from entity %05zu named %s to its EntityGroup: %s",
        eid, entityName, GxfResultStr(code));
      GXF_LOG_ERROR("Deactivating...");
      auto result = deactivate();
      if (!result) { GXF_LOG_ERROR("Deactivation failed."); }
      return Unexpected{code};
    }
  }
  return Success;
}

Expected<void> Program::preDeactivateEntities(const FixedVector<Entity, kMaxEntities>& entities) {
  for (size_t i = 0; i < entities.size(); i++) {
    const gxf_uid_t eid = entities.at(i)->eid();
    const gxf_result_t code = entity_warden_->depopulateResourcesFromEntityGroup(context_, eid);
    if (code != GXF_SUCCESS) {
      GXF_LOG_ERROR("Failed to remove resources in entity [eid: %05zu] from its EntityGroup: %s",
                    eid, GxfResultStr(code));
      return Unexpected{code};
    }
  }
  return Success;
}

Expected<void> Program::scheduleEntity(gxf_uid_t eid) {
  std::lock_guard<std::recursive_mutex> lock(entity_mutex_);

  if (state_ == State::ORIGIN) {
    GXF_LOG_ERROR("Graph must be activated before activating individual entities");
    return Unexpected{GXF_INVALID_LIFECYCLE_STAGE};
  }

  auto maybe = Entity::Shared(context_, eid);
  if (!maybe) { return ForwardError(maybe); }
  auto entity = maybe.value();

  // Check if entity was previously added to program and is yet to be scheduled
  bool is_schedulable = false;
  for (size_t i = 0; i < unscheduled_entities_.size(); ++i) {
    if (unscheduled_entities_.at(i).value().eid() == eid) {
      unscheduled_entities_.erase(i);
      is_schedulable = true;
      break;
    }
  }

  if (!is_schedulable) { return Success; }

  // Find all systems
  auto systems = entity.findAllHeap<System>();
  if (!systems) {
    return ForwardError(systems);
  }
  for (auto system : systems.value()) {
    if (!system) {
      GXF_LOG_ERROR("Found a bad system component while scheduling entity %s", entity.name());
      return Unexpected{GXF_FAILURE};
    }
    auto result = system_group_->addSystem(std::move(system.value()));
    if (!result) { return ForwardError(result); }
  }

  // Offer entity executor to all the schedulers
  auto schedulers = entity.findAllHeap<Scheduler>();
  if (!schedulers) { return ForwardError(schedulers); }
  for (auto scheduler : schedulers.value()) {
    if (!scheduler) {
      GXF_LOG_ERROR("Found a bad scheduler component while scheduling entity %s", entity.name());
      return Unexpected{GXF_FAILURE};
    }

    const gxf_result_t code = scheduler.value()->prepare_abi(entity_executor_);
    if (code != GXF_SUCCESS) { return Unexpected{code}; }
  }
  // Keep track of entities with schedulers
  if (!schedulers->empty()) { schedulers_.insert(eid); }

  // Find all monitors
  auto monitors = entity.findAllHeap<Monitor>();
  if (!monitors) {
    return ForwardError(monitors);
  }
  for (auto monitor : monitors.value()) {
    if (!monitor) {
      GXF_LOG_ERROR("Found a bad monitor component while scheduling entity %s", entity.name());
      return Unexpected{GXF_FAILURE};
    }
    auto result = entity_executor_->addMonitor(monitor.value());
    if (!result) { return ForwardError(result); }
  }

  // Find all statistics
  auto job_statistics = entity.findAllHeap<JobStatistics>();
  if (!job_statistics) {
    return ForwardError(job_statistics);
  }
  for (auto statistics : job_statistics.value()) {
    if (!statistics) {
      GXF_LOG_ERROR("Found a bad job statistics component while scheduling entity %s",
                    entity.name());
      return Unexpected{GXF_FAILURE};
    }
    auto result = entity_executor_->addStatistics(statistics.value());
    if (!result) { return ForwardError(result); }
  }

  // Find all IPC servers
  auto ipc_servers = entity.findAllHeap<IPCServer>();
  if (ipc_servers) {
    for (auto server : ipc_servers.value()) {
      server.value()->registerService(IPCServer::Service{
        "config",
        IPCServer::kAction,
        {
          nullptr,
          std::bind(&Program::onParameterSet, this, std::placeholders::_1, std::placeholders::_2)
        }
      });
      server.value()->registerService(IPCServer::Service{
        "dump",
        IPCServer::kQuery,
        {
          std::bind(&Program::onGraphDump, this, std::placeholders::_1),
          nullptr
        }
      });
    }
  }

  scheduled_entities_.emplace_back(entity);

  // Pass it to system group for execution
  return system_group_->schedule(std::move(entity));
}

Expected<void> Program::unscheduleEntity(gxf_uid_t eid) {
  std::lock_guard<std::recursive_mutex> lock(entity_mutex_);
  auto maybe = Entity::Shared(context_, eid);
  if (!maybe) { return ForwardError(maybe); }
  auto entity = maybe.value();

  for (size_t i = 0; i < scheduled_entities_.size(); ++i) {
    if (scheduled_entities_.at(i).value().eid() == eid) {
      // unschedule it from the system group
      auto result = system_group_->unschedule(entity);
      scheduled_entities_.erase(i);
      unscheduled_entities_.emplace_back(entity);
      if (!result) { return result; }
      break;
    }
  }

  // Remove all statistics
  auto job_statistics = entity.findAllHeap<JobStatistics>();
  if (!job_statistics) {
    return ForwardError(job_statistics);
  }
  for (auto statistics : job_statistics.value()) {
    if (!statistics) {
      GXF_LOG_ERROR("Found a bad job statistics component while unscheduling entity %s",
                    entity.name());
      return Unexpected{GXF_FAILURE};
    }
    auto result = entity_executor_->removeStatistics(statistics.value());
    if (!result) { return ForwardError(result); }
  }

  // Remove all monitors
  auto monitors = entity.findAllHeap<Monitor>();
  if (!monitors) {
    return ForwardError(monitors);
  }
  for (auto monitor : monitors.value()) {
    if (!monitor) {
      GXF_LOG_ERROR("Found a bad monitor component while unscheduling entity %s", entity.name());
      return Unexpected{GXF_FAILURE};
    }
    auto result = entity_executor_->removeMonitor(monitor.value());
    if (!result) { return ForwardError(result); }
  }

  // Remove if it's a scheduler entity
  if (schedulers_.find(eid) != schedulers_.end()) {
    schedulers_.erase(schedulers_.find(eid));
  }

  // Un-register entity with all routers
  const Expected<void> result = router_group_->removeRoutes(entity);
  if (!result) { return ForwardError(result); }

  // Remove all routers
  auto routers = entity.findAllHeap<Router>();
  if (!routers) {
    return ForwardError(routers);
  }
  for (auto router : routers.value()) {
    if (!router) {
      GXF_LOG_ERROR("Found a bad router component while unscheduling entity %s", entity.name());
      return Unexpected{GXF_FAILURE};
    }
    auto result = router_group_->removeRouter(router.value());
    if (!result) { return ForwardError(result); }
  }

  // Remove all systems
  auto systems = entity.findAllHeap<System>();
  if (!systems) {
    return ForwardError(systems);
  }
  for (auto system : systems.value()) {
    if (!system) {
      GXF_LOG_ERROR("Found a bad system component while unscheduling entity %s", entity.name());
      return Unexpected{GXF_FAILURE};
    }
    auto result = system_group_->removeSystem(std::move(system.value()));
    if (!result) { return ForwardError(result); }
  }

  return Success;
}

Expected<void> Program::runAsync() {
  State activated = State::ACTIVATED;
  if (!state_.compare_exchange_strong(activated, State::STARTING)) {
    GXF_LOG_ERROR("Unexpected State: %hhd", static_cast<int32_t>(state_.load()));
    return Unexpected{GXF_INVALID_EXECUTION_SEQUENCE};
  }

  if (system_group_->empty()) {
    GXF_LOG_WARNING("No GXF scheduler specified.");
  }

  const Expected<void> result = system_group_->runAsync();
  if (!result) {
    GXF_LOG_ERROR("Couldn't run async. Deactivating...");
    auto code = deactivate();
    if (!code) { GXF_LOG_ERROR("Deactivation failed."); }
    return ForwardError(result);
  }

  state_ = State::RUNNING;

  return Success;
}

Expected<void> Program::interrupt() {
  State running = State::RUNNING;
  if (!state_.compare_exchange_strong(running, State::INTERRUPTING)) {
    GXF_LOG_ERROR("Attempted interrupting when not running (state=%hhd).",
                  static_cast<int32_t>(state_.load()));
    return Unexpected{GXF_INVALID_EXECUTION_SEQUENCE};
  }

  return system_group_->stop();
}

Expected<void> Program::wait() {
  State state = state_.load();

  // Graph already in a stopped state.
  if (state == State::ACTIVATED || state == State::ORIGIN || state == State::DEINITALIZING) {
    return Success;
  }

  if (state != State::STARTING && state != State::RUNNING && state != State::INTERRUPTING) {
    GXF_LOG_ERROR("Unexpected State: %hhd", static_cast<int32_t>(state_.load()));
    return Unexpected{GXF_INVALID_EXECUTION_SEQUENCE};
  }

  const Expected<void> result = system_group_->wait();
  if (!result) {
    GXF_LOG_ERROR("wait failed. Deactivating...");
    auto code = deactivate();
    if (!code) { GXF_LOG_ERROR("Deactivation failed."); }
    return ForwardError(result);
  }

  // State could be: RUNNING - voluntary stop by scheduler.
  //                 INTERRUPTING - instructed stop by API.
  //                 <OTHER> - state updated by other thread waited on the runtime.
  state = state_.load();
  if (state == State::RUNNING || state == State::INTERRUPTING) {
    state_.compare_exchange_strong(state, State::ACTIVATED);
  }

  return Success;
}

Expected<void> Program::entityEventNotify(gxf_uid_t eid, gxf_event_t event) {
  if (!system_group_) { return Success; }

  State state = state_.load();

  // Ignore the event if notified during deinitialization
  if (state == State::DEINITALIZING || state == State::ACTIVATING) {
    const char* entityName = "UNKNOWN";
    GxfEntityGetName(context_, eid, &entityName);
    GXF_LOG_DEBUG("Ignoring event notification for entity [%s] with id [%ld] since graph is"
                   " [%s]", entityName, eid, programStateStr(state));
    return Success;
  }

  if (state != State::RUNNING && state != State::INTERRUPTING && state != State::STARTING) {
    const char* entityName = "UNKNOWN";
    GxfEntityGetName(context_, eid, &entityName);
    GXF_LOG_ERROR("Event notification %d for entity [%s] with id [%ld] received in an unexpected "
                  "state [%s]", event, entityName, eid, programStateStr(state));
    return Unexpected{GXF_INVALID_EXECUTION_SEQUENCE};
  }

  const auto result = system_group_->event_notify(eid, event);
  if (!result) {
    return ForwardError(result);
  }

  return Success;
}

Expected<void> Program::deactivate() {
  if (state_ == State::ORIGIN) { return Success; }

  state_ = State::DEINITALIZING;

  // Make a copy of all the graph eids
  FixedVector<gxf_uid_t, kMaxEntities> graph_eids;

  for (size_t i = 0; i < unscheduled_entities_.size(); ++i) {
    auto eid = unscheduled_entities_.at(i).value().eid();
    if (schedulers_.find(eid) != schedulers_.end()) { continue; }
    auto result = graph_eids.emplace_back(eid);
    if (!result) {
      resetProgram();
      return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE};
    }
  }

  for (size_t i = 0; i < scheduled_entities_.size(); ++i) {
    auto eid = scheduled_entities_.at(i).value().eid();
    if (schedulers_.find(eid) != schedulers_.end()) { continue; }
    auto result = graph_eids.emplace_back(eid);
    if (!result) {
      resetProgram();
      return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE};
    }
  }

  // Deactivate all graph entities first (scheduled and unscheduled)
  // Deactivating the entities in the reverse order.
  for (int32_t i = (static_cast<int32_t>(graph_eids.size())) - 1; i >= 0; i--) {
    const gxf_result_t code = GxfEntityDeactivate(context_, graph_eids.at(i).value());
    if (code != GXF_SUCCESS) {
      resetProgram();
      return Unexpected{code};
    }
  }

  // Make a copy of the scheduler eids
  FixedVector<gxf_uid_t, kMaxEntities> scheduler_eids;
  for (const auto& scheduler : schedulers_) {
    auto result = scheduler_eids.emplace_back(scheduler);
    if (!result) {
      resetProgram();
      return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE};
    }
  }

  // Deactivate all the scheduler entities next
  for (size_t i = 0; i < scheduler_eids.size(); i++) {
    const gxf_result_t code = GxfEntityDeactivate(context_, scheduler_eids.at(i).value());
    if (code != GXF_SUCCESS) {
      resetProgram();
      return Unexpected{code};
    }
  }

  router_group_entity_ = Entity();
  system_group_entity_ = Entity();

  state_ = State::ORIGIN;
  return Success;
}

Expected<void> Program::destroy() {
  unscheduled_entities_.clear();
  scheduled_entities_.clear();
  schedulers_.clear();

  return Success;
}

Expected<void> Program::onParameterSet(const std::string& resource, const std::string& data) {
  auto end = resource.find("/");
  if (end == std::string::npos) {
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  // to locate the parameter, both name and id is supported:
  // name: "entity_name/component name/key"
  // id;   "cid/key"
  std::vector<std::string> tokens;
  std::string::size_type begin = 0;
  for (size_t end = 0; (end = resource.find("/", end)) != std::string::npos; ++end) {
    tokens.push_back(resource.substr(begin, end-begin));
    begin = end+1;
  }
  tokens.push_back(resource.substr(begin));

  auto value = YAML::Load(data);
  gxf_uid_t cid = kNullUid;
  std::string key;
  if (tokens.size() == 2) {
    // using cid/key
    try {
       cid = std::stoull(tokens[0]);
    } catch (std::invalid_argument&) {
      return Unexpected{GXF_ARGUMENT_INVALID};
    }

    key = tokens[1];
  } else if (tokens.size() == 3) {
    //  using entity/component/key
    gxf_uid_t eid = kNullUid;
    gxf_result_t result = GxfEntityFind(context_, tokens[0].c_str(), &eid);
    if (GXF_SUCCESS != result) {
      return Unexpected{result};
    }
    result = GxfComponentFind(context_, eid, GxfTidNull(), tokens[1].c_str(), nullptr, &cid);
    if (GXF_SUCCESS != result) {
      return Unexpected{result};
    }
    key = tokens[2];
  }

  return parameter_storage_->parse(cid, key.c_str(), value, "");
}

Expected<std::string> Program::onGraphDump(const std::string& resource) {
  if (resource == "*") {
    // dump the whole graph
    return dumpGraph(kUnspecifiedUid);
  } else {
    try {
      gxf_uid_t uid = std::stoll(resource);
      return dumpGraph(uid);
    } catch (std::invalid_argument &) {
      return Unexpected{GXF_INVALID_DATA_FORMAT};
    }
  }
  return Unexpected{GXF_ARGUMENT_INVALID};
}

Expected<std::string> Program::dumpGraph(gxf_uid_t uid) {
  YAML::Emitter out;
  bool filter_eid = false;
  State state = state_.load();
  FixedVector<Entity> entity_list;
  if (state == State::ORIGIN) {
    entity_list.reserve(unscheduled_entities_.size());
    entity_list.copy_from(unscheduled_entities_);
  } else {
    entity_list.reserve(scheduled_entities_.size());
    entity_list.copy_from(scheduled_entities_);
  }

  if (uid != kUnspecifiedUid) {
    for (auto e : entity_list) {
      if (e.value().eid() == uid) {
        filter_eid = true;
        break;
      }
    }
  }
  for (auto e : entity_list) {
    const gxf_uid_t eid = e.value().eid();
    if (filter_eid && uid != kUnspecifiedUid && uid != eid) {
      continue;
    }

    const char *entity_name;
    const gxf_result_t result_2 = GxfEntityGetName(context_, eid, &entity_name);
    if (result_2 != GXF_SUCCESS) {
      GXF_LOG_ERROR("Could not get name for the entity E%05zu", eid);
      return Unexpected{result_2};
    }
    // Get an entity's components
    gxf_uid_t cids[kMaxComponents];
    uint64_t num_cids = kMaxComponents;
    const gxf_result_t result_3 = GxfComponentFindAll(context_, eid, &num_cids, cids);
    if (result_3 != GXF_SUCCESS) {
      GXF_LOG_ERROR(
        "Could not find all components for the entity %s (E%05zu)",
        entity_name, eid);
      return Unexpected{result_3};
    }
    // Skip empty entities
    if (num_cids == 0) {
      continue;
    }
    // Skip unwanted entities when filtering by cid
    if (uid != kUnspecifiedUid &&!filter_eid) {
      bool has_cid = false;
      for (size_t i = 0; i < num_cids; i++) {
        if (cids[i] == uid) {
          has_cid = true;
          break;
        }
      }
      if (!has_cid) {
        continue;
      }
    }

    out << YAML::BeginDoc;
    out << YAML::BeginMap;
    if (std::strcmp(entity_name, "") != 0) {
      out << YAML::Key << "name";
      out << YAML::Value << entity_name;
    }
    out << YAML::Key << "id";
    out << YAML::Value << eid;
    // Start exporting an entity's components
    out << YAML::Key << "components";
    out << YAML::Value;
    out << YAML::BeginSeq;

    // Export each component
    for (uint64_t j=0; j < num_cids; j++) {
      gxf_uid_t cid = cids[j];
      if (!filter_eid && uid != kUnspecifiedUid && uid != cid) {
        continue;
      }
      out << YAML::BeginMap;

      // Component name
      const char *comp_name;
      const gxf_result_t result_4 = GxfComponentName(context_, cid, &comp_name);
      if (result_4 != GXF_SUCCESS) {
        GXF_LOG_ERROR(
          "Could not get name for component C%05zu in entity %s (E%05zu)",
          cid, entity_name, eid);
        return Unexpected{result_4};
      }
      // A component's name can be empty. Export only if it's not empty.
      if (std::strcmp(comp_name, "") != 0) {
        out << YAML::Key << "name";
        out << YAML::Value << comp_name;
        out << YAML::Key << "id";
        out << YAML::Value << cid;
      }

      // Component type
      gxf_tid_t tid;
      const char *comp_type_name;
      const gxf_result_t result_5 = GxfComponentType(context_, cid, &tid);
      if (result_5 != GXF_SUCCESS) {
        GXF_LOG_ERROR("Could not get type for component %s/%s (C%05zu)",
          entity_name, comp_name, cid);
        return Unexpected{result_5};
      }
      const gxf_result_t result_6 = GxfComponentTypeName(context_, tid, &comp_type_name);
      if (result_6 != GXF_SUCCESS) {
        GXF_LOG_ERROR("Could not get name for component type %016lx%016lx",
          tid.hash1, tid.hash2);
        return Unexpected{result_6};
      }
      out << YAML::Key << "type";
      out << YAML::Value << comp_type_name;

      // Component parameters
      out << YAML::Key << "parameters";
      out << YAML::Value;
      out << YAML::BeginMap;

      gxf_component_info_t comp_info;
      const char *parameter_name_ptrs[kMaxParameters];
      comp_info.parameters = parameter_name_ptrs;
      comp_info.num_parameters = kMaxParameters;
      const gxf_result_t result_7 = GxfComponentInfo(context_, tid, &comp_info);
      if (result_7 != GXF_SUCCESS) {
        GXF_LOG_ERROR("Could not get info for component type %016lx%016lx",
          tid.hash1, tid.hash2);
        return Unexpected{result_7};
      }

      // Export each parameter
      for (uint64_t parameter_i=0; parameter_i < comp_info.num_parameters; parameter_i++) {
        gxf_parameter_info_t param_info;
        const gxf_result_t result_8 = GxfGetParameterInfo(
          context_, tid, comp_info.parameters[parameter_i], &param_info);
        if (result_8 != GXF_SUCCESS) {
          GXF_LOG_ERROR("Could not get parameter info for component type %016lx%016lx",
            tid.hash1, tid.hash2);
          return Unexpected{result_8};
        }

        auto maybe = parameter_storage_->wrap(cid, param_info.key);
        if (maybe) {
          out << YAML::Key << param_info.key;
          out << YAML::Value << maybe.value();
        } else if (maybe.error() != GXF_PARAMETER_NOT_INITIALIZED) {
          GXF_LOG_ERROR("Failed to wrap parameter %s with error %s", param_info.key,
                        GxfResultStr(maybe.error()));
          return Unexpected{GXF_FAILURE};
        }
      }
      // Close map for a component's parameters
      out << YAML::EndMap;
      // Close map for the component
      out << YAML::EndMap;
    }
    // Close seq for the component list in an entity
    out << YAML::EndSeq;
    // Close map for an entity
    out << YAML::EndMap;
  }
  out << YAML::EndDoc;
  return out.c_str();
}

const char* Program::programStateStr(const State& state) {
  switch (state) {
    GXF_ENUM_TO_STR(State::ORIGIN, Origin)
    GXF_ENUM_TO_STR(State::ACTIVATING, Activating)
    GXF_ENUM_TO_STR(State::ACTIVATED, Activated)
    GXF_ENUM_TO_STR(State::STARTING, Starting)
    GXF_ENUM_TO_STR(State::RUNNING, Running)
    GXF_ENUM_TO_STR(State::INTERRUPTING, Interrupting)
    GXF_ENUM_TO_STR(State::DEINITALIZING, Deinitializing)
    default:
      return "N/A";
  }
}

void Program::resetProgram() {
  router_group_entity_ = Entity();
  system_group_entity_ = Entity();
  scheduled_entities_.clear();
  unscheduled_entities_.clear();
}

}  // namespace gxf
}  // namespace nvidia
