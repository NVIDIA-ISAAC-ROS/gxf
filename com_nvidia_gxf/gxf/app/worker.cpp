/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gxf/app/worker.hpp"

#include <string>

#include "gxf/app/application.hpp"
#include "gxf/ipc/http/http_ipc_client.hpp"
#include "gxf/ipc/http/http_server.hpp"

namespace nvidia {
namespace gxf {
/**
 * Worker representation in Application class
*/
Worker::Worker(Application* owner, const std::string& name) {
  if (owner == nullptr) {
    GXF_LOG_ERROR("Failed to create Application's Worker,"
      "by providing invalid Application pointer");
    return;
  }
  owner_ = owner;
  name_ = name;
  worker_entity_ = owner_->createGraphEntity("WorkerEntity_" + name_);
  graph_worker_ = worker_entity_->add<GraphWorker>(name.c_str());
  server_ = worker_entity_->add<HttpServer>("worker_ipc_server");
  client_ = worker_entity_->add<HttpIPCClient>("worker_ipc_client");
}

Expected<void> Worker::commit() {
  RETURN_IF_ERROR(server_->setParameter(kRemoteAccess, true));
  if (port_ > 0) {
    RETURN_IF_ERROR(server_->setParameter(kPort, port_));
  }
  if (!driver_ip_.empty()) {
    RETURN_IF_ERROR(client_->setParameter(kServerIpAddress, driver_ip_));
  }
  if (driver_port_ > 0) {
    RETURN_IF_ERROR(client_->setParameter(kPort, driver_port_));
  }
  RETURN_IF_ERROR(graph_worker_->setParameter("server", server_));
  RETURN_IF_ERROR(graph_worker_->setParameter("client", client_));
  if (this->segment_names_.empty()) {
    GXF_LOG_ERROR("No segments selected for GraphWorker[name: %s]", name_.c_str());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  for (const auto& segment_name : this->segment_names_) {
    auto it = owner_->segments_.find(segment_name);
    if (it == owner_->segments_.end()) {
      GXF_LOG_ERROR("Selected Segment[name: %s] to GraphWorker[name: %s] is not found",
        segment_name.c_str(), name_.c_str());
    } else {
      GXF_LOG_INFO("Add Segment[name: %s] to GraphWorker[name: %s]",
        it->first.c_str(), name_.c_str());
      /**
       * Key interface how GraphWorker takes over the run of C++ API Segments
      */
      graph_worker_->addSegment(it->first, it->second->context());
    }
  }
  return Success;
}

}  // namespace gxf
}  // namespace nvidia
