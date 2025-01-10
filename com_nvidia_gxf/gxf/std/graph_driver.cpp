/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <memory>
#include <string>
#include <vector>

#include "gxf/std/graph_driver.hpp"
#include "gxf/std/graph_driver_worker_common.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t GraphDriver::schedule_abi(gxf_uid_t eid) {
  return GXF_SUCCESS;
}

gxf_result_t GraphDriver::unschedule_abi(gxf_uid_t eid) {
  return GXF_SUCCESS;
}

gxf_result_t GraphDriver::runAsync_abi() {
  // instantiate flow control object
  driver_thread_ = std::make_unique<GxfSystemThread>(
    std::bind(&GraphDriver::asyncRunnerCallback, this, std::placeholders::_1, this),
    name());
  return GXF_SUCCESS;
}

gxf_result_t GraphDriver::stop_abi() {
  driver_thread_->stop();
  return GXF_SUCCESS;
}

gxf_result_t GraphDriver::wait_abi() {
  driver_thread_->wait();
  return GXF_SUCCESS;
}

gxf_result_t GraphDriver::event_notify_abi(gxf_uid_t eid, gxf_event_t event) {
  return GXF_SUCCESS;
}

gxf_result_t GraphDriver::registerInterface(Registrar* registrar) {
  nvidia::gxf::Expected<void> result;
  // This ECS component layer
  result &= registrar->parameter(
    list_of_connections_, "connections", "Connection between different graphs",
    "A dictionary of source and target graphs, "
    "{Connections: [source:"", target:""]}", {});

  // Communication layer
  result &= registrar->parameter(server_, "server", "API server", "");
  result &= registrar->parameter(client_, "client", "API client", "");
  return nvidia::gxf::ToResultCode(result);
}

gxf_result_t GraphDriver::initialize() {
  if (!server_.try_get()) {
    GXF_LOG_ERROR("%s: 'server' parameter not set", name());
    return GXF_FAILURE;
  }

  if (!client_.try_get()) {
    GXF_LOG_ERROR("%s: 'client' parameter not set", name());
    return GXF_FAILURE;
  }

  // Connections
  // source: segment_name.entity_name.tx_name
  // target: segment_name.entity_name.rx_name
  if (connections_.empty()) {
    GXF_LOG_INFO("GraphDriver YAML API flow, need to obtain segment connections from YAML spec");
    if (list_of_connections_.get().empty()) {
      GXF_LOG_ERROR("list of connections not set");
      return GXF_FAILURE;
    }
    for (const auto& connection : list_of_connections_.get()) {
      auto source = connection.at("source");
      auto target = connection.at("target");
      connections_[source] = target;
      reverse_connections_[target] = source;

      // Populate segment names
      auto end = source.find(".");
      if (end == std::string::npos) {
        return GXF_ARGUMENT_INVALID;
      }
      segment_names_.insert(source.substr(0, end));

      end = target.find(".");
      if (end == std::string::npos) {
        return GXF_ARGUMENT_INVALID;
      }
      segment_names_.insert(target.substr(0, end));
    }
  } else {
    GXF_LOG_INFO("GraphDriver C++ API flow, "
      "obtained %ld segment connections from addSegmentConnection() API",
      connections_.size());
    if (connections_.size() != reverse_connections_.size()) {
      std::stringstream ss;
      for (const auto& name : segment_names_) {
        ss << name << ", ";
      }
      GXF_LOG_ERROR("GraphDriver got mismatched segment connections. "
        "connections[%ld], reverse_connections[%ld], segment_names[%s]",
        connections_.size(), reverse_connections_.size(), ss.str().c_str());
    }
  }

  auto maybe_server = server_.try_get();
  if (maybe_server) {
    auto server_handle = maybe_server.value();
    {  // register execute segments service
      IPCServer::Service assign_worker_address = {
        "RegisterGraphWorker",
        IPCServer::ServiceType::kAction,
        {.action = std::bind(&GraphDriver::onRegisterGraphWorker, this,
          std::placeholders::_1, std::placeholders::_2)}
      };
      Expected<void> result = server_handle->registerService(assign_worker_address);
      if (!result) {
        return result.error();
      }
      // GXF_LOG_INFO("Start serving execute segments service at uri: %s",
      //   execute_segments_uri_.try_get().value().c_str());
    }
    {  // register execute segments service
      IPCServer::Service worker_complete = {
        "GraphWorkerComplete",
        IPCServer::ServiceType::kAction,
        {.action = std::bind(&GraphDriver::onGraphWorkerComplete, this,
          std::placeholders::_1, std::placeholders::_2)}
      };
      Expected<void> result = server_handle->registerService(worker_complete);
      if (!result) {
        return result.error();
      }
    }
  }

  return GXF_SUCCESS;
}

bool GraphDriver::asyncRunnerCallback(std::string event, GraphDriver* self) {
  // GXF use Expected<> to replace exception,
  // however thread callback has to add exception catch, the same to std::thread entrance
  try {
    GXF_LOG_DEBUG("[%s] GraphDriver thread received event: %s", self->name(), event.c_str());
    Expected<void> result = Unexpected{GXF_FAILURE};
    if (event == Event::kResolveConnections) {
      result = self->resolveConnections();
    } else if (event == Event::kExecuteWorkers) {
      result = self->executeWorkers();
    } else if (event == Event::kDeactivateWorkers) {
      result = self->deactivateWorkers();
    } else if (event == Event::kStopWorkers) {
      result = self->stopWorkers();
      GXF_LOG_INFO("Stopping GraphDriver");
      return false;  // unblock wait() thread
    } else {
      // an error occurred, clean up if needed then exit
      GXF_LOG_ERROR("Unknown event: %s", event.c_str());
      result = Unexpected{GXF_FAILURE};
    }
    // an error occurred, clean up if needed then exit
    if (!result) {
      GXF_LOG_ERROR("GraphWorker:%s unexpected error in asyncRunnerCallback: %s",
        name(), result.get_error_message());
      return false;  // unblock wait() thread
    }

    return true;
  } catch (...) {
    GXF_LOG_ERROR("GraphDriver:%s unexpected error in asyncRunnerCallback.", name());
    return false;  // unblock wait() thread
  }
}

Expected<void> GraphDriver::executeWorkers() {
  for (auto worker : workers_) {
    std::string result;
    std::string ip_address;
    int port = -1;
    parseIpAddress(worker.first, ip_address, port);
    GXF_LOG_INFO("ActivateSegments on GraphWorker: [%s:%d]", ip_address.c_str(), port);
    auto res = client_->changeAddress(ip_address, port)
      .action("ActivateSegments", "GraphWorker", "placeholder");
    if (!res) {
      GXF_LOG_ERROR("ActivateSegments on GraphWorker: %s failed!", worker.first.c_str());
      return res;
    }
    GXF_LOG_INFO("RunSegments on GraphWorker: [%s:%d]", ip_address.c_str(), port);
    res = client_->action("RunSegments", "GraphWorker", "placeholder");
    if (!res) {
      GXF_LOG_ERROR("RunSegments on GraphWorker: %s failed!", worker.first.c_str());
      return res;
    }
  }
  return Success;
}

Expected<void> GraphDriver::deactivateWorkers() {
  for (auto worker : workers_) {
    GXF_LOG_INFO("deactivateWorkers() on GraphWorker: %s", worker.first.c_str());
    std::string result;
    std::string ip_address;
    int port = -1;
    parseIpAddress(worker.first, ip_address, port);
    GXF_LOG_INFO("DeactivateSegments on GraphWorker: [%s:%d]", ip_address.c_str(), port);
    auto res = client_->changeAddress(ip_address, port)
      .action("DeactivateSegments", "GraphWorker", "placeholder");
    if (!res) {
      GXF_LOG_ERROR("DeactivateSegments on GraphWorker: %s failed!", worker.first.c_str());
    }
    GXF_LOG_INFO("DestroySegments on GraphWorker: [%s:%d]", ip_address.c_str(), port);
    res = client_->action("DestroySegments", "GraphWorker", "placeholder");
    if (!res) {
      GXF_LOG_ERROR("DestroySegments on GraphWorker: %s failed!", worker.first.c_str());
    }
  }
  return Success;
}

Expected<void> GraphDriver::stopWorkers() {
  for (auto worker : workers_) {
    GXF_LOG_INFO("stopWorkers() on GraphWorker: %s", worker.first.c_str());
    std::string result;
    std::string ip_address;
    int port = -1;
    parseIpAddress(worker.first, ip_address, port);
    GXF_LOG_INFO("StopWorker on GraphWorker: [%s:%d]", ip_address.c_str(), port);
    auto res = client_->changeAddress(ip_address, port)
      .action("StopWorker", "GraphWorker", "placeholder");
    if (!res) {
      GXF_LOG_ERROR("StopWorker on GraphWorker: %s failed!", worker.first.c_str());
    }
  }
  return Success;
}

Expected<void> GraphDriver::resolveConnections() {
  GXF_LOG_DEBUG("Start resolveConnections ...");
  // std::vector<ComponentParam> params;
  for (auto& worker : workers_) {
    std::vector<ComponentParam> params;
    auto segment_names = worker.second;
    auto worker_address = worker.first;

    for (auto& connection : connections_) {
      ComponentParam param;
      std::vector<ComponentParam::ParamInfo> param_infos;
      auto tx = connection.first;
      auto rx = connection.second;
      std::string tx_segment_name, tx_entity_name, tx_component_name;
      std::string rx_segment_name, rx_entity_name, rx_component_name;
      parseSegmentEntityComponentName(tx, tx_segment_name, tx_entity_name, tx_component_name);
      parseSegmentEntityComponentName(rx, rx_segment_name, rx_entity_name, rx_component_name);

      auto it = std::find(segment_names.begin(), segment_names.end(), tx_segment_name);
      if (it != segment_names.end()) {
        param.segment_name = tx_segment_name;
        param.entity_name = tx_entity_name;
        param.component_name = tx_component_name;
        auto ip_port = segment_ip_address_[rx];
        std::string ip_address;
        int port = -1;
        parseIpAddress(ip_port, ip_address, port);
        // Set ip address
        ComponentParam::ParamInfo param_info_ip_address;
        param_info_ip_address.value_type = ParameterTypeTrait<std::string>::type_name;
        param_info_ip_address.key = "receiver_address";
        param_info_ip_address.value = ip_address;
        param_infos.push_back(param_info_ip_address);
        // Set port
        ComponentParam::ParamInfo param_info_port;
        param_info_port.value_type = ParameterTypeTrait<uint32_t>::type_name;
        param_info_port.key = "port";
        param_info_port.value = std::to_string(port);
        param_infos.push_back(param_info_port);
        param.params = param_infos;
        GXF_LOG_DEBUG("Prepare payload to set UCX Tx[%s] parameters:\n"
          " key: %s, value: %s, value_type: %s,\n"
          " key: %s, value: %s, value_type: %s\n",
          param.serialize().c_str(),
          param_info_ip_address.key.c_str(),
          param_info_ip_address.value.c_str(),
          param_info_ip_address.value_type.c_str(),
          param_info_port.key.c_str(),
          param_info_port.value.c_str(),
          param_info_port.value_type.c_str());
        params.push_back(param);
      }
    }
    std::string result;
    std::string ip_address;
    int port = -1;
    parseIpAddress(worker_address, ip_address, port);

    if (params.empty()) {
      GXF_LOG_INFO("Skip SetComponentParams on GraphWorker: [%s:%d]", ip_address.c_str(), port);
      continue;
    }

    auto serialized_string = GraphDriverWorkerParser::serialize_onSetComponentParams(params);
    GXF_LOG_INFO("SetComponentParams on GraphWorker: [%s:%d]", ip_address.c_str(), port);
    auto res = client_->changeAddress(ip_address, port)
      .action("SetComponentParams", "GraphWorker", serialized_string.value());
    if (!res) {
      GXF_LOG_ERROR("Send request to GraphWorker: %s failed!", worker_address.c_str());
    }
  }

  GXF_LOG_INFO("Finish resolveConnections, start executeWorkers ...");
  driver_thread_->queueItem(Event::kExecuteWorkers);
  return Success;
}

Expected<void> GraphDriver::onRegisterGraphWorker(
  const std::string& resource, const std::string& payload
) {
  GXF_LOG_DEBUG("Start onRegisterGraphWorker() with payload: %s",
    payload.c_str());
  auto result = GraphDriverWorkerParser::deserialize_onRegisterGraphWorker(payload);
  if (!result) {
    GXF_LOG_ERROR("Deserializing on Graph Worker failed!");
    return Unexpected{GXF_FAILURE};
  }
  auto worker_info = result.value();
  GXF_LOG_INFO("Connection request from GraphWorker: [%s] received",
    worker_info.ip_port().c_str());
  std::vector<std::string> segment_names;
  // Check if segment name is valid or not
  for (const auto& segment : worker_info.segment_info_list) {
    if (segment_names_.find(segment.segment_name) == segment_names_.end()) {
      GXF_LOG_ERROR("Segment %s not valid", segment.segment_name.c_str());
      return Unexpected{GXF_ARGUMENT_INVALID};
    } else if (requested_segment_names_.find(segment.segment_name) !=
      requested_segment_names_.end()) {
      GXF_LOG_ERROR("Segment %s already requested in another GraphWorker",
        segment.segment_name.c_str());
      return Unexpected{GXF_ARGUMENT_INVALID};
    } else {
      requested_segment_names_.insert(segment.segment_name);
      segment_names.push_back(segment.segment_name);
      for (auto& pair : segment.ip_port_address_map) {
        GXF_LOG_DEBUG("From GraphWorker received UCX Rx[%s] IP:PORT %s",
          pair.first.c_str(), pair.second.c_str());
        segment_ip_address_[pair.first] = pair.second;
      }
    }
  }
  // Add GraphWorker ip address and port with segment names
  workers_[worker_info.server_ip_address + ":" + worker_info.server_port] = segment_names;
  if (requested_segment_names_.size() == segment_names_.size()) {
    GXF_LOG_INFO("Received all GraphWorker registration, progress[%ld / %ld]."
      " Proceed to next stage ResulveConnections.",
      requested_segment_names_.size(), segment_names_.size());
    driver_thread_->queueItem(Event::kResolveConnections);
  } else {
    GXF_LOG_INFO("Received GraphWorker registration progress[%ld / %ld]."
      " Waiting for more connection", requested_segment_names_.size(), segment_names_.size());
  }

  return Success;
}

Expected<void> GraphDriver::onGraphWorkerComplete(
  const std::string& resource, const std::string& payload
) {
  GXF_LOG_DEBUG("Start onGraphWorkerComplete() with payload: %s",
    payload.c_str());
  auto result = GraphDriverWorkerParser::deserialize_onRegisterGraphWorker(payload);
  if (!result) {
    GXF_LOG_ERROR("Deserializing on GraphWorker failed!");
    return Unexpected{GXF_FAILURE};
  }
  auto worker_info = result.value();
  GXF_LOG_INFO("Complete notification from GraphWorker: %s received",
    worker_info.server_ip_address.c_str());

  auto it = workers_.find(worker_info.ip_port());
  if (it == workers_.end()) {
    GXF_LOG_ERROR("Completed GraphWorker not registered at beginning");
    return Unexpected{GXF_FAILURE};
  } else {
    completed_workers_.emplace(worker_info.ip_port());
    if (completed_workers_.size() < workers_.size()) {
      GXF_LOG_INFO("GraphWorkers complete in progress: %ld / %ld",
        completed_workers_.size(), workers_.size());
    } else if (completed_workers_.size() == workers_.size()) {
      GXF_LOG_INFO("All registered GraphWorkers completed. "
        "Deactivating and stopping all GraphWorkers");
      driver_thread_->queueItem(Event::kDeactivateWorkers);
      driver_thread_->queueItem(Event::kStopWorkers);
    } else {
      GXF_LOG_ERROR("Number of completed GraphWorkers larger than registered GraphWorkers");
      return Unexpected{GXF_FAILURE};
    }
  }

  return Success;
}

Expected<void> GraphDriver::addSegmentConnection(const std::string& source,
  const std::string& target
) {
  connections_[source] = target;
  reverse_connections_[target] = source;

  // Populate segment names
  auto end = source.find(".");
  if (end == std::string::npos) {
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  segment_names_.insert(source.substr(0, end));

  end = target.find(".");
  if (end == std::string::npos) {
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  segment_names_.insert(target.substr(0, end));
  return Success;
}

}  // namespace gxf
}  // namespace nvidia
