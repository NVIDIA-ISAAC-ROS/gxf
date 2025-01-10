/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/graph_worker.hpp"

// header for getPrimaryIp()
#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
// getPrimaryIp()

#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nvidia {
namespace gxf {

gxf_result_t GraphWorker::schedule_abi(gxf_uid_t eid) {
  return GXF_SUCCESS;
}

gxf_result_t GraphWorker::unschedule_abi(gxf_uid_t eid) {
  return GXF_SUCCESS;
}

gxf_result_t GraphWorker::runAsync_abi() {
  // instantiate flow control object
  worker_thread_ = std::make_unique<GxfSystemThread>(
    std::bind(&GraphWorker::asyncRunnerCallback, this, std::placeholders::_1, this),
    name());

  if (segment_runners_.empty()) {
    GXF_LOG_INFO("GraphWorker YAML API flow, populating gxf context from spec and manifest");
    auto future1 = worker_thread_->queueItem(Event::kInstantiateSegmentRunner);
    if (!future1.get()) {
      GXF_LOG_ERROR("Failed to instantiate segment runner");
      return GXF_FAILURE;
    }
  } else {
    GXF_LOG_INFO("GraphWorker C++ API flow, taking populated context from Segment");
  }
  auto future2 = worker_thread_->queueItem(Event::kRegisterWorker);
  if (!future2.get()) {
    GXF_LOG_ERROR("Failed to register GraphWorker[name: %s]", name());
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t GraphWorker::stop_all_segments() {
  GXF_LOG_INFO("Stopping all segment threads");
  for (const auto& pair : segment_runners_) {
    pair.second->stop();
  }
  return GXF_SUCCESS;
}

gxf_result_t GraphWorker::stop_abi() {
  GXF_LOG_INFO("[%s] GraphWorker stopping...", name());
  // stop all SegmentRunners thread
  stop_all_segments();
  // stop GraphWorker thread
  worker_thread_->stop();
  return GXF_SUCCESS;
}

gxf_result_t GraphWorker::wait_abi() {
  GXF_LOG_DEBUG("segment_runners_.size() %ld", segment_runners_.size());
  // block on joining each SegmentRunner thread
  for (const auto& pair : segment_runners_) {
    GXF_LOG_DEBUG("SegmentRunner[%s] wait()", pair.first.c_str());
    pair.second->wait();
  }
  // block on joining GraphWorker's thread
  GXF_LOG_DEBUG("GraphWorker[%s] wait()", name());
  worker_thread_->wait();
  return GXF_SUCCESS;
}

gxf_result_t GraphWorker::event_notify_abi(gxf_uid_t eid, gxf_event_t event) {
  return GXF_SUCCESS;
}

gxf_result_t GraphWorker::registerInterface(Registrar* registrar) {
  nvidia::gxf::Expected<void> result;
  // This ECS component layer
  result &= registrar->parameter(
    graph_specs_, "graph-specs", "Graph spec paths",
    "A dictionary of graph specs, "
    "{name1: {app-path: , parameter-path: , manifest-path: , severity: }}",
    {});
  result &= registrar->parameter(driver_reconnection_times_,
    "driver-reconnection-times", "Driver Reconnection Times",
    "How many times to try to connect driver", 3l);

  // Communication layer
  result &= registrar->parameter(server_, "server", "API server",
    "API Server to handle service callbacks");
  result &= registrar->parameter(client_, "client", "API client",
    "API Client to request driver server");

  // Service layer
  result &= registrar->parameter(initialize_segments_uri_, "initialize-segments-uri",
    "Initialize segments uri",
    "Customize service uri of activating segments",
    std::string("InitializeSegments"), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(set_component_params_uri_, "set-component-params-uri",
    "Set component params uri",
    "Customize service uri of setting component parameters",
    std::string("SetComponentParams"), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(activate_segments_uri_, "activate-segments-uri",
    "Activate segments uri",
    "Customize service uri of activating segments",
    std::string("ActivateSegments"), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(run_segments_uri_, "run-segments-uri",
    "Run segments uri",
    "Customize service uri of executing all segments specified in this worker",
    std::string("RunSegments"), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(deactivate_segments_uri_, "deactivate-segments-uri",
    "Deactivate segments uri",
    "Customize service uri of deactivating segments",
    std::string("DeactivateSegments"), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(destroy_segments_uri_, "destroy-segments-uri",
    "Destroy segments uri",
    "Customize service uri of destroying segments",
    std::string("DestroySegments"), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(stop_worker_uri_, "stop-worker-uri",
    "Stop worker uri",
    "Customize service uri of stopping worker and all its segments",
    std::string("StopWorker"), GXF_PARAMETER_FLAGS_OPTIONAL);

  return nvidia::gxf::ToResultCode(result);
}

gxf_result_t GraphWorker::initialize() {
  if (!initialize_segments_uri_.try_get() || initialize_segments_uri_.try_get().value().empty()) {
    GXF_LOG_ERROR("%s: 'initialize_segments_uri_' should be non empty string",
            name());
    return GXF_FAILURE;
  }
  if (!set_component_params_uri_.try_get() || set_component_params_uri_.try_get().value().empty()) {
    GXF_LOG_ERROR("%s: 'set_component_params_uri_' should be non empty string",
            name());
    return GXF_FAILURE;
  }
  if (!activate_segments_uri_.try_get() || activate_segments_uri_.try_get().value().empty()) {
    GXF_LOG_ERROR("%s: 'set_component_params_uri_' should be non empty string",
            name());
    return GXF_FAILURE;
  }
  if (!run_segments_uri_.try_get() || run_segments_uri_.try_get().value().empty()) {
    GXF_LOG_ERROR("%s: 'run_segments_uri_' should be non empty string",
            name());
    return GXF_FAILURE;
  }
  if (!deactivate_segments_uri_.try_get() || deactivate_segments_uri_.try_get().value().empty()) {
    GXF_LOG_ERROR("%s: 'deactivate_segments_uri_' should be non empty string",
            name());
    return GXF_FAILURE;
  }
  if (!destroy_segments_uri_.try_get() || destroy_segments_uri_.try_get().value().empty()) {
    GXF_LOG_ERROR("%s: 'destroy_segments_uri_' should be non empty string",
            name());
    return GXF_FAILURE;
  }
  if (!stop_worker_uri_.try_get() || stop_worker_uri_.try_get().value().empty()) {
    GXF_LOG_ERROR("%s: 'stop_worker_uri_' should be non empty string",
            name());
    return GXF_FAILURE;
  }

  { // register Initialize segments service
    IPCServer::Service service_initialize_segments = {
      initialize_segments_uri_.try_get().value(),
      IPCServer::ServiceType::kAction,
      {.action = std::bind(&GraphWorker::onInitializeSegments, this,
        std::placeholders::_1, std::placeholders::_2)}
    };
    Expected<void> result = server_.get()->registerService(service_initialize_segments);
    if (!result) {
      GXF_LOG_ERROR("Failed to register service Initialize segments");
      return result.error();
    }
    GXF_LOG_INFO("Start serving Initialize segments service at uri: %s",
      initialize_segments_uri_.try_get().value().c_str());
  }
  { // register Activate segments service
    IPCServer::Service service_activate_segments = {
      activate_segments_uri_.try_get().value(),
      IPCServer::ServiceType::kAction,
      {.action = std::bind(&GraphWorker::onActivateSegments, this,
        std::placeholders::_1, std::placeholders::_2)}
    };
    Expected<void> result = server_.get()->registerService(service_activate_segments);
    if (!result) {
      GXF_LOG_ERROR("Failed to register service Activate segments");
      return result.error();
    }
    GXF_LOG_INFO("Start serving Activate segments service at uri: %s",
      activate_segments_uri_.try_get().value().c_str());
  }
  { // register Run segments service
    IPCServer::Service service_run_segments = {
      run_segments_uri_.try_get().value(),
      IPCServer::ServiceType::kAction,
      {.action = std::bind(&GraphWorker::onRunSegments, this,
        std::placeholders::_1, std::placeholders::_2)}
    };
    Expected<void> result = server_.get()->registerService(service_run_segments);
    if (!result) {
      GXF_LOG_ERROR("Failed to register service Run segments");
      return result.error();
    }
    GXF_LOG_INFO("Start serving Run segments service at uri: %s",
      run_segments_uri_.try_get().value().c_str());
  }
  { // register deactivate_segments service
    IPCServer::Service service_deactivate_segments = {
      deactivate_segments_uri_.try_get().value(),
      IPCServer::ServiceType::kAction,
      {.action = std::bind(&GraphWorker::onDeactivateSegments, this,
        std::placeholders::_1, std::placeholders::_2)}
    };
    Expected<void> result = server_.get()->registerService(service_deactivate_segments);
    if (!result) {
      GXF_LOG_ERROR("Failed to register service deactivate_segments");
      return result.error();
    }
    GXF_LOG_INFO("Start serving deactivate_segments service at uri: %s",
      deactivate_segments_uri_.try_get().value().c_str());
  }
  { // register destroy_segments service
    IPCServer::Service service_destroy_segments = {
      destroy_segments_uri_.try_get().value(),
      IPCServer::ServiceType::kAction,
      {.action = std::bind(&GraphWorker::onDestroySegments, this,
        std::placeholders::_1, std::placeholders::_2)}
    };
    Expected<void> result = server_.get()->registerService(service_destroy_segments);
    if (!result) {
      GXF_LOG_ERROR("Failed to register service destroy_segments");
      return result.error();
    }
    GXF_LOG_INFO("Start serving destroy_segments service at uri: %s",
      destroy_segments_uri_.try_get().value().c_str());
  }
  { // register exit worker service
    IPCServer::Service service_stop_worker = {
      stop_worker_uri_.try_get().value(),
      IPCServer::ServiceType::kAction,
      {.action = std::bind(&GraphWorker::onStopWorker, this,
        std::placeholders::_1, std::placeholders::_2)}
    };
    Expected<void> result = server_.get()->registerService(service_stop_worker);
    if (!result) {
      GXF_LOG_ERROR("Failed to register service service_stop_worker");
      return result.error();
    }
    GXF_LOG_INFO("Start serving service_stop_worker service at uri: %s",
      stop_worker_uri_.try_get().value().c_str());
  }
  { // register set component params service
    IPCServer::Service service_set_component_params = {
      set_component_params_uri_.try_get().value(),
      IPCServer::ServiceType::kAction,
      {.action = std::bind(&GraphWorker::onSetComponentParams, this,
        std::placeholders::_1, std::placeholders::_2)}
    };
    Expected<void> result = server_.get()->registerService(service_set_component_params);
    if (!result) {
      GXF_LOG_ERROR("Failed to register service set component params");
      return result.error();
    }
    GXF_LOG_INFO("Start serving service set component params at uri: %s",
      set_component_params_uri_.try_get().value().c_str());
  }

  return GXF_SUCCESS;
}

gxf_result_t GraphWorker::deinitialize() {
  Expected<void> result = Success;
  return ToResultCode(result);
}

Expected<std::string> GraphWorker::createWorkerInfo() {
  if (worker_info_ == nullptr) {
    worker_info_ = std::make_unique<WorkerInfo>();
    worker_info_->server_ip_address = getPrimaryIp();
    uint32_t port;
    gxf_result_t result = GxfParameterGetUInt32(context(),
      server_.get()->cid(), "port", &port);
    if (result != GXF_SUCCESS) {
      GXF_LOG_ERROR("Failed to get param of IPCServer port");
      return Unexpected{result};
    }

    worker_info_->server_port = std::to_string(port);
    GXF_LOG_DEBUG("GraphWorker server cid %ld, server_port %s",
      server_.get()->cid(), worker_info_->server_port.c_str());
    for (const auto& seg : segment_runners_) {
      auto seg_info = seg.second->createSegmentInfo(worker_info_->server_ip_address);
      if (!seg_info) {
        GXF_LOG_ERROR("Failed to create segment info for segment: %s",
          worker_info_->server_ip_address.c_str());
        return ForwardError(seg_info);
      }
      worker_info_->segment_info_list.emplace_back(seg_info.value());
    }
  }

  return GraphDriverWorkerParser::serialize_onRegisterGraphWorker(*worker_info_);
}

Expected<void> GraphWorker::registerGraphWorker() {
  auto payload = createWorkerInfo();
  if (!payload) {
    GXF_LOG_ERROR("Failed to create serialized WorkerInfo");
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  int64_t retry_times = 0;
  int64_t retry_duration = 1;
  Expected<void> result = Unexpected{GXF_FAILURE};
  while (!result && retry_times <= driver_reconnection_times_.get()) {
    result = client_.get()->action("RegisterGraphWorker", "GraphDriver", payload.value());
    if (!result) {
      GXF_LOG_ERROR("Failed IPC request to GraphDriver, payload: %s",
        payload.value().c_str());
      std::this_thread::sleep_for(std::chrono::seconds(retry_duration++));
      retry_times++;
    }
  }
  return result;
}

Expected<void> GraphWorker::instantiateSegmentRunners() {
  for (const auto& pair : graph_specs_.get()) {
    const std::string name = pair.first;
    const GraphSpec spec = pair.second;
    auto segment_runner_ptr = std::make_unique<SegmentRunner>(name, spec,
      this->worker_thread_);
    // create gxf context, load manifest, load graph
    std::future<bool> future = segment_runner_ptr->asyncInitializeGxfGraph();
    if (!future.get()) {
      GXF_LOG_ERROR("Failed to instantiate segment runner %s", name.c_str());
      return Unexpected{GXF_FAILURE};
    }
    segment_runners_.emplace(name, std::move(segment_runner_ptr));
  }
  return Success;
}

Expected<void> GraphWorker::checkComplete() {
  const size_t segment_runner_nums = segment_runners_.size();
  completed_segment_runner_nums_++;
  if (completed_segment_runner_nums_ < segment_runner_nums) {
    GXF_LOG_DEBUG("segment-runner-complete event received: %ld / %ld",
      completed_segment_runner_nums_, segment_runner_nums);
  } else if (segment_runner_nums == 0 || completed_segment_runner_nums_ == segment_runner_nums) {
    GXF_LOG_DEBUG("All segment-runner-complete event received: %ld / %ld",
      completed_segment_runner_nums_, segment_runner_nums);
    GXF_LOG_INFO("Reporting complete to remote GraphDriver...");
    auto payload = createWorkerInfo();
    if (!payload) {
      GXF_LOG_ERROR("Failed to create serialized WorkerInfo");
      return Unexpected{payload.error()};
    } else {
      auto result = client_.get()->action("GraphWorkerComplete", "GraphDriver", payload.value());
      if (!result) {
        GXF_LOG_ERROR("Failed IPC request to GraphDriver, payload: %s",
          payload.value().c_str());
        return result;
      }
    }
  } else {
    GXF_LOG_ERROR("More segment-runner-complete event received than actual segment runners");
    return Unexpected{GXF_FAILURE};
  }
  return Success;
}

bool GraphWorker::asyncRunnerCallback(std::string event, GraphWorker* self) {
  // GXF use Expected<> to replace exception,
  // however thread callback has to add exception catch, the same to std::thread entrance
  try {
    GXF_LOG_DEBUG("[%s] GraphWorker thread received event: %s", self->name(), event.c_str());
    Expected<void> result = Unexpected{GXF_FAILURE};
    if (event == Event::kInstantiateSegmentRunner) {
      result = self->instantiateSegmentRunners();
    } else if (event == Event::kRegisterWorker) {
      result = self->registerGraphWorker();
    } else if (event == Event::kCheckWorkComplete) {
      result = self->checkComplete();
    } else {
      GXF_LOG_ERROR("Unknown event: %s", event.c_str());
      result = Unexpected{GXF_FAILURE};
    }
    // an error occurred, clean up if needed then exit
    if (!result) {
      GXF_LOG_ERROR("GraphWorker:%s unexpected error in asyncRunnerCallback: %s",
        name(), result.get_error_message());
      stop_all_segments();
      return false;
    }

    return true;
  } catch (...) {
    GXF_LOG_ERROR("GraphWorker:%s unexpected error in asyncRunnerCallback.", name());
    stop_all_segments();
    return false;
  }
}

Expected<void> GraphWorker::onInitializeSegments(
  const std::string& resource, const std::string& payload) {
  for (const auto& seg_ptr : segment_runners_) {
    GXF_LOG_INFO("Initializing graph segment %s", seg_ptr.first.c_str());
    std::future<bool> future = seg_ptr.second->asyncInitializeGxfGraph();
    if (!future.get()) {
      GXF_LOG_ERROR("Failed to initiailze GXF graph in segment %s",
        seg_ptr.first.c_str());
      return Unexpected{GXF_FAILURE};
    }
  }
  return Success;
}

Expected<void> GraphWorker::onActivateSegments(
  const std::string& resource, const std::string& payload) {
  for (const auto& seg_ptr : segment_runners_) {
    GXF_LOG_INFO("Activating graph segment %s", seg_ptr.first.c_str());
    std::future<bool> future = seg_ptr.second->asyncActivateGxfGraph();
    /**
     * we can optionally sync the activate. Since there is one single thread to handle
     * enqueued events, the event execution is sync'ed by default following calling order
     * future.get();
    */
  }
  return Success;
}

Expected<void> GraphWorker::onRunSegments(
  const std::string& resource, const std::string& payload) {
  Expected<void> result;
  if (segment_runners_.empty()) {
    this->worker_thread_->queueItem(GraphWorker::Event::kCheckWorkComplete);
  } else {
    for (const auto& seg_ptr : segment_runners_) {
      GXF_LOG_INFO("Starting graph segment %s", seg_ptr.first.c_str());
      seg_ptr.second->runGxfGraph();
    }
  }
  return result;
}

Expected<void> GraphWorker::onDeactivateSegments(
  const std::string& resource, const std::string& payload) {
  for (const auto& seg_ptr : segment_runners_) {
    GXF_LOG_INFO("Deactivating graph segment %s", seg_ptr.first.c_str());
    std::future<bool> future = seg_ptr.second->asyncDeactivateGxfGraph();
    if (!future.get()) {
      GXF_LOG_ERROR("Failed to deactivating GXF graph in segment %s",
        seg_ptr.first.c_str());
      return Unexpected{GXF_FAILURE};
    }
  }
  return Success;
}

Expected<void> GraphWorker::onDestroySegments(
  const std::string& resource, const std::string& payload) {
  for (const auto& seg_ptr : segment_runners_) {
    GXF_LOG_INFO("Destroy graph segment %s", seg_ptr.first.c_str());
    std::future<bool> future = seg_ptr.second->asyncDestroyGxfGraph();
    if (!future.get()) {
      GXF_LOG_ERROR("Failed to destroy GXF graph in segment %s",
        seg_ptr.first.c_str());
      return Unexpected{GXF_FAILURE};
    }
  }
  return Success;
}

Expected<void> GraphWorker::onStopWorker(
  const std::string& resource, const std::string& payload) {
  gxf_result_t code = this->stop_abi();
  if (code != GXF_SUCCESS) { return Unexpected{code}; }
  return Success;
}

Expected<void> GraphWorker::onSetComponentParams(
  const std::string& resource, const std::string& payload) {
  Expected<void> result;
  Expected<std::vector<ComponentParam>> component_param_list =
    GraphDriverWorkerParser::deserialize_onSetComponentParams(payload);
  if (!component_param_list) {
    GXF_LOG_ERROR("Failed to parse payload: %s", payload.c_str());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  for (const auto& comp : component_param_list.value()) {
    auto it = segment_runners_.find(comp.segment_name);
    if (it != segment_runners_.end()) {
      for (const auto& param : comp.params) {
        it->second->setParameter(comp.entity_name, comp.component_name,
          param.key, param.value, param.value_type);
      }
    } else {
      GXF_LOG_ERROR("[%s] GraphWorker doesn't have graph segment %s",
        name(), comp.segment_name.c_str());
      result = Unexpected{GXF_ARGUMENT_INVALID};
    }
  }
  return result;
}

// TODO(byin): can improve readability by moving to another file
std::string GraphWorker::getPrimaryIp() {
  int fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd == -1) {
    GXF_LOG_ERROR("Cannot create socket");
    return "";
  }

  struct ifconf ifc;
  char buf[1024];
  ifc.ifc_len = sizeof(buf);
  ifc.ifc_buf = buf;
  if (ioctl(fd, SIOCGIFCONF, &ifc) == -1) {
    GXF_LOG_ERROR("ioctl error");
    close(fd);
    return "";
  }

  struct ifreq* it = ifc.ifc_req;
  const struct ifreq* const end = it + (ifc.ifc_len / sizeof(struct ifreq));
  std::string ip;

  for (; it != end; ++it) {
    if (ioctl(fd, SIOCGIFFLAGS, it) == 0) {
      if (!(it->ifr_flags & IFF_LOOPBACK)) {  // don't count loopback
        if (ioctl(fd, SIOCGIFADDR, it) == 0) {
          ip = inet_ntoa(((struct sockaddr_in*)&it->ifr_addr)->sin_addr);
          break;
        }
      }
    } else {
      GXF_LOG_ERROR("ioctl error");
      close(fd);
      return "";
    }
  }

  close(fd);
  return ip;
}

Expected<void> GraphWorker::addSegment(const std::string& name,
  const gxf_context_t context
) {
  if (segment_runners_.find(name) != segment_runners_.end()) {
    GXF_LOG_ERROR("Segment[%s] already added to GraphWorker[%s]",
      name.c_str(), this->name());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  if (context == kNullContext) {
    GXF_LOG_ERROR("Cannot add Segment[%s] with kNullContext to GraphWorker[%s]",
      name.c_str(), this->name());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  // expect a gxf context that already done: create context, load manifest, load graph
  auto segment_runner_ptr = std::make_unique<SegmentRunner>(name, context, this->worker_thread_);
  segment_runners_.emplace(name, std::move(segment_runner_ptr));

  return Success;
}

/**
 * SegmentRunner
*/
void SegmentRunner::wait() {
  runner_thread_->wait();
}

void SegmentRunner::stop() {
  runner_thread_->stop();
}

std::future<bool> SegmentRunner::asyncInitializeGxfGraph() {
  runner_thread_->queueItem(Event::kCreateGxfContext);
  runner_thread_->queueItem(Event::kLoadGxfManifest);
  return runner_thread_->queueItem(Event::kLoadGxfGraph);
}

std::future<bool> SegmentRunner::asyncActivateGxfGraph() {
  return runner_thread_->queueItem(Event::kActivateGxfGraph);
}

std::future<bool> SegmentRunner::asyncRunGxfGraph() {
  return runner_thread_->queueItem(Event::kNonBlockingRunGxfGraph);
}

std::future<bool> SegmentRunner::runGxfGraph() {
  return runner_thread_->queueItem(Event::kBlockingRunGxfGraph);
}

std::future<bool> SegmentRunner::asyncDeactivateGxfGraph() {
  return runner_thread_->queueItem(Event::kDeactivateGxfGraph);
}

std::future<bool> SegmentRunner::asyncDestroyGxfGraph() {
  return runner_thread_->queueItem(Event::kDestroyGxfGraph);
}

bool SegmentRunner::asyncRunnerCallback(std::string event, SegmentRunner* self) {
  if (event == Event::kCreateGxfContext) {
    std::unique_lock<std::mutex> lock(context_mutex_);
    self->wrapCreateGxfConext();
  } else if (event == Event::kLoadGxfManifest) {
    std::unique_lock<std::mutex> lock(context_mutex_);
    self->wrapLoadGxfManifest();
  } else if (event == Event::kLoadGxfGraph) {
    std::unique_lock<std::mutex> lock(context_mutex_);
    self->wrapLoadGxfGraph();
  } else if (event == Event::kActivateGxfGraph) {
    std::unique_lock<std::mutex> lock(context_mutex_);
    self->wrapActivateGxfGraph();
  } else if (event == Event::kNonBlockingRunGxfGraph) {
    std::unique_lock<std::mutex> lock(context_mutex_);
    self->wrapNonBlockingRunGxfGraph();
  } else if (event == Event::kBlockingRunGxfGraph) {
    std::unique_lock<std::mutex> lock(context_mutex_);
    self->wrapBlockingRunGxfGraph();
    self->worker_thread_->queueItem(GraphWorker::Event::kCheckWorkComplete);
  } else if (event == Event::kInterruptGxfGraph) {
    std::unique_lock<std::mutex> lock(context_mutex_);
    self->wrapInterruptGxfGraph();
  } else if (event == Event::kDeactivateGxfGraph) {
    std::unique_lock<std::mutex> lock(context_mutex_);
    self->wrapDeactivateGxfGraph();
  } else if (event == Event::kDestroyGxfGraph) {
    std::unique_lock<std::mutex> lock(context_mutex_);
    self->wrapDestroyGxfGraph();
  } else {
    // an error occurred, clean up if needed then exit
    GXF_LOG_ERROR("Unknown event: %s", event.c_str());
  }
  return true;
}

Expected<SegmentInfo> SegmentRunner::createSegmentInfo(const std::string& worker_host_ip) {
  SegmentInfo segment_info;
  segment_info.segment_name = this->name_;
  gxf_result_t code;

  gxf_tid_t tid;
  code = GxfComponentTypeId(context_, "nvidia::gxf::UcxReceiver", &tid);
  if (code != GXF_SUCCESS) {
    if (code == GXF_FACTORY_UNKNOWN_CLASS_NAME) {
      GXF_LOG_WARNING("Graph segment[%s] has no nvidia::gxf::UcxReceiver. "
        "Return empty SegmentInfo", this->name_.c_str());
      return segment_info;
    }
    return Unexpected{code};
  }

  gxf_uid_t entities[kMaxEntities];
  uint64_t num_entities = kMaxEntities;
  code = GxfEntityFindAll(context_, &num_entities, entities);
  if (code != GXF_SUCCESS) { return Unexpected{code}; }
  std::vector<gxf_uid_t> cid_list;
  for (uint64_t i = 0; i < num_entities; i++) {
    const gxf_uid_t eid = entities[i];
    for (int offset = 0; ; offset++) {
      gxf_uid_t cid;
      code = GxfComponentFind(context_, eid, tid, nullptr, &offset, &cid);
      if (code != GXF_SUCCESS) { break; }
      // each ucx rx component
      cid_list.push_back(cid);
      // entity name
      const char* entity_name;
      code = GxfEntityGetName(context_, eid, &entity_name);
      if (code != GXF_SUCCESS) { return Unexpected{code}; }
      // component name
      const char* component_name;
      code = GxfComponentName(context_, cid, &component_name);
      ComponentInfo ucx_rx{segment_info.segment_name, entity_name, component_name};
      // TODO(byin): smart NIC IP detect to replace reading from config
      uint32_t value_port;
      gxf_result_t result = GxfParameterGetUInt32(context_, cid, "port", &value_port);
      if (result != GXF_SUCCESS) {
        return Unexpected{result};
      }
      const std::string ip_port = worker_host_ip + ":" + std::to_string(value_port);
      GXF_LOG_DEBUG("UCX Rx[%s] resolves its IP:PORT as [%s]",
        ucx_rx.serialize().c_str(), ip_port.c_str());
      segment_info.ip_port_address_map.emplace(ucx_rx.serialize(), ip_port);
    }
  }
  return segment_info;
}

gxf_result_t SegmentRunner::setParameter(
  const std::string& entity_name, const std::string& comp_name,
  const std::string& key, const std::string& value, const std::string& value_type
) {
  gxf_result_t code;
  gxf_uid_t eid;
  gxf_uid_t cid;
  std::unique_lock<std::mutex> lock(context_mutex_);
  code = GxfEntityFind(context_, entity_name.c_str(), &eid);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("[%s] GxfEntityFind Error: %s", name_.c_str(), GxfResultStr(code));
    return code;
  }
  code = GxfComponentFind(context_, eid, ucx_transmitter_tid, comp_name.c_str(), nullptr, &cid);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("[%s] GxfComponentFind Error: %s", name_.c_str(), GxfResultStr(code));
    return code;
  }
  if (value_type == ParameterTypeTrait<bool>::type_name) {
    auto converted_value = ComponentParam::ParamInfo::strToBool(value);
    if (!converted_value) {
      GXF_LOG_ERROR("Component[%s] parameter[key %s, value %s] is not a valid type of %s",
        comp_name.c_str(), key.c_str(), value.c_str(), value_type.c_str());
      return GXF_ARGUMENT_INVALID;
    }
    code = GxfParameterSetBool(context_, cid, key.c_str(), converted_value.value());
  } else if (value_type == ParameterTypeTrait<float>::type_name) {
    auto converted_value = ComponentParam::ParamInfo::strToFloat32(value);
    if (!converted_value) {
      GXF_LOG_ERROR("Component[%s] parameter[key %s, value %s] is not a valid type of %s",
        comp_name.c_str(), key.c_str(), value.c_str(), value_type.c_str());
      return GXF_ARGUMENT_INVALID;
    }
    code = GxfParameterSetFloat32(context_, cid, key.c_str(), converted_value.value());
  } else if (value_type == ParameterTypeTrait<double>::type_name) {
    auto converted_value = ComponentParam::ParamInfo::strToFloat64(value);
    if (!converted_value) {
      GXF_LOG_ERROR("Component[%s] parameter[key %s, value %s] is not a valid type of %s",
        comp_name.c_str(), key.c_str(), value.c_str(), value_type.c_str());
      return GXF_ARGUMENT_INVALID;
    }
    code = GxfParameterSetInt64(context_, cid, key.c_str(), converted_value.value());
  } else if (value_type == ParameterTypeTrait<uint16_t>::type_name) {
    auto converted_value = ComponentParam::ParamInfo::strToUInt16(value);
    if (!converted_value) {
      GXF_LOG_ERROR("Component[%s] parameter[key %s, value %s] is not a valid type of %s",
        comp_name.c_str(), key.c_str(), value.c_str(), value_type.c_str());
      return GXF_ARGUMENT_INVALID;
    }
    code = GxfParameterSetUInt16(context_, cid, key.c_str(), converted_value.value());
  } else if (value_type == ParameterTypeTrait<int32_t>::type_name) {
    auto converted_value = ComponentParam::ParamInfo::strToInt32(value);
    if (!converted_value) {
      GXF_LOG_ERROR("Component[%s] parameter[key %s, value %s] is not a valid type of %s",
        comp_name.c_str(), key.c_str(), value.c_str(), value_type.c_str());
      return GXF_ARGUMENT_INVALID;
    }
    code = GxfParameterSetInt32(context_, cid, key.c_str(), converted_value.value());
  } else if (value_type == ParameterTypeTrait<int64_t>::type_name) {
    auto converted_value = ComponentParam::ParamInfo::strToInt64(value);
    if (!converted_value) {
      GXF_LOG_ERROR("Component[%s] parameter[key %s, value %s] is not a valid type of %s",
        comp_name.c_str(), key.c_str(), value.c_str(), value_type.c_str());
      return GXF_ARGUMENT_INVALID;
    }
    code = GxfParameterSetInt64(context_, cid, key.c_str(), converted_value.value());
  } else if (value_type == ParameterTypeTrait<uint32_t>::type_name) {
    auto converted_value = ComponentParam::ParamInfo::strToUInt32(value);
    if (!converted_value) {
      GXF_LOG_ERROR("Component[%s] parameter[key %s, value %s] is not a valid type of %s",
        comp_name.c_str(), key.c_str(), value.c_str(), value_type.c_str());
      return GXF_ARGUMENT_INVALID;
    }
    code = GxfParameterSetUInt32(context_, cid, key.c_str(), converted_value.value());
  } else if (value_type == ParameterTypeTrait<uint64_t>::type_name) {
    auto converted_value = ComponentParam::ParamInfo::strToUInt64(value);
    if (!converted_value) {
      GXF_LOG_ERROR("Component[%s] parameter[key %s, value %s] is not a valid type of %s",
        comp_name.c_str(), key.c_str(), value.c_str(), value_type.c_str());
      return GXF_ARGUMENT_INVALID;
    }
    code = GxfParameterSetUInt64(context_, cid, key.c_str(), converted_value.value());
  } else if (value_type == ParameterTypeTrait<std::string>::type_name) {
    code = GxfParameterSetStr(context_, cid, key.c_str(), value.c_str());
  } else {
    GXF_LOG_ERROR("Set type[%s] is not supported for "
      "component[%s] parameter[key %s, value %s]",
      value_type.c_str(), comp_name.c_str(), key.c_str(), value.c_str());
    return GXF_ARGUMENT_INVALID;
  }
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("[%s] GxfParameterSet%s Error: %s",
      name_.c_str(), value_type.c_str(), GxfResultStr(code));
    return code;
  }

  GXF_LOG_INFO("Successfully set gxf::Component parameter:\n"
    " Graph segment: %s, Entity name: %s, Component name: %s\n"
    " key: %s, value: %s, value_type: %s",
    name_.c_str(), entity_name.c_str(), comp_name.c_str(),
    key.c_str(), value.c_str(), value_type.c_str());
  return GXF_SUCCESS;
}

// gxf wrappers, non thread safe to gxf_context
gxf_result_t SegmentRunner::wrapCreateGxfConext() {
  if (graph_spec_.severity < 0 || graph_spec_.severity > 4) {
    GXF_LOG_WARNING("[%s] invalid severity: %d", name_.c_str(),
      graph_spec_.severity);
  }
  gxf_result_t code;
  code = GxfContextCreate(&context_);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("[%s] GxfContextCreate Error: %s", name_.c_str(), GxfResultStr(code));
    return code;
  }
  s_signal_context_ = context_;
  GXF_LOG_INFO("[%s] Setting severity: %d", name_.c_str(), graph_spec_.severity);
  code = GxfSetSeverity(context_, static_cast<gxf_severity_t>(graph_spec_.severity));
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("[%s] GxfSetSeverity Error: %s", name_.c_str(), GxfResultStr(code));
    return code;
  }
  return GXF_SUCCESS;
}

gxf_result_t SegmentRunner::wrapLoadGxfManifest() {
  if (graph_spec_.manifest_path.empty()) {
    GXF_LOG_ERROR("[%s] empty manifest path", name_.c_str());
    return GXF_ARGUMENT_INVALID;
  }
  gxf_result_t code;
  GXF_LOG_INFO("[%s] Loading manifest: '%s'", name_.c_str(), graph_spec_.manifest_path.c_str());
  const char* manifest[] = {graph_spec_.manifest_path.c_str()};
  const GxfLoadExtensionsInfo info{nullptr, 0, manifest, 1, nullptr};
  code = GxfLoadExtensions(context_, &info);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("[%s] GxfLoadExtensions Error: %s", name_.c_str(), GxfResultStr(code));
    return code;
  }
  return GXF_SUCCESS;
}

gxf_result_t SegmentRunner::wrapLoadGxfGraph() {
  if (graph_spec_.app_path.empty()) {
    GXF_LOG_ERROR("[%s] empty app path", name_.c_str());
    return GXF_ARGUMENT_INVALID;
  }
  gxf_result_t code;
  GXF_LOG_INFO("[%s] Loading app: '%s'", name_.c_str(), graph_spec_.app_path.c_str());
  code = GxfGraphLoadFile(context_, graph_spec_.app_path.c_str(), nullptr, 0U);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("[%s] GxfGraphLoadFile Error: %s", name_.c_str(), GxfResultStr(code));
    return code;
  }
  return GXF_SUCCESS;
}

gxf_result_t SegmentRunner::wrapActivateGxfGraph() {
  gxf_result_t code;
  GXF_LOG_INFO("Activating %s ...", name_.c_str());
  code = GxfGraphActivate(context_);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("[%s] GxfGraphActivate Error: %s", name_.c_str(), GxfResultStr(code));
    return code;
  }
  return GXF_SUCCESS;
}

gxf_result_t SegmentRunner::wrapBlockingRunGxfGraph() {
  gxf_result_t code;
  GXF_LOG_INFO("Running %s ...", name_.c_str());
  code = GxfGraphRunAsync(context_);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("[%s] GxfGraphRunAsync Error: %s", name_.c_str(), GxfResultStr(code));
    return code;
  }

  // block current thread
  code = GxfGraphWait(context_);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("[%s] GxfGraphWait Error: %s", name_.c_str(), GxfResultStr(code));
    return code;
  }
  return GXF_SUCCESS;
}

gxf_result_t SegmentRunner::wrapNonBlockingRunGxfGraph() {
  gxf_result_t code;
  GXF_LOG_INFO("Running %s ...", name_.c_str());
  code = GxfGraphRunAsync(context_);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("[%s] GxfGraphRunAsync Error: %s", name_.c_str(), GxfResultStr(code));
    return code;
  }
  return GXF_SUCCESS;
}

gxf_result_t SegmentRunner::wrapInterruptGxfGraph() {
  gxf_result_t code = GxfGraphInterrupt(s_signal_context_);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("GxfGraphInterrupt Error: %s", GxfResultStr(code));
  }
  return GXF_SUCCESS;
}

gxf_result_t SegmentRunner::wrapDeactivateGxfGraph() {
  gxf_result_t code;
  GXF_LOG_INFO("Deinitializing...");
  code = GxfGraphDeactivate(context_);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("[%s] GxfGraphDeactivate Error: %s", name_.c_str(), GxfResultStr(code));
    return code;
  }
  GXF_LOG_INFO("[%s] Context deinitialized.", name_.c_str());
  return GXF_SUCCESS;
}

gxf_result_t SegmentRunner::wrapDestroyGxfGraph() {
  gxf_result_t code;
  GXF_LOG_INFO("Destroying context...");
  code = GxfContextDestroy(context_);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("[%s] GxfContextDestroy Error: %s", name_.c_str(), GxfResultStr(code));
    return code;
  }
  GXF_LOG_INFO("[%s] Context destroyed.", name_.c_str());
  return GXF_SUCCESS;
}
//
// end SegmentRunner
//

}  // namespace gxf
}  // namespace nvidia
