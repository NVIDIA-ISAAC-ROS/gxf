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

#include "gxf/app/driver.hpp"

#include <string>

#include "gxf/app/application.hpp"
#include "gxf/ipc/http/http_ipc_client.hpp"
#include "gxf/ipc/http/http_server.hpp"

namespace nvidia {
namespace gxf {
/**
 * Driver representation in Application class
*/
Driver::Driver(Application* owner, const std::string& name) {
  if (owner == nullptr) {
    GXF_LOG_ERROR("Failed to create Application's Driver, "
      "by providing invalid Application pointer");
    return;
  }
  owner_ = owner;
  name_ = name;
  driver_entity_ = owner_->createGraphEntity("DriverEntity_" + name_);
  graph_driver_ = driver_entity_->add<GraphDriver>(name_.c_str());
  server_ = driver_entity_->add<HttpServer>("driver_ipc_server");
  client_ = driver_entity_->add<HttpIPCClient>("driver_ipc_client");
}

Expected<void> Driver::commit() {
  RETURN_IF_ERROR(server_->setParameter(kRemoteAccess, true));
  if (port_ > 0) {
    RETURN_IF_ERROR(server_->setParameter(kPort, port_));
  }
  RETURN_IF_ERROR(graph_driver_->setParameter("server", server_));
  RETURN_IF_ERROR(graph_driver_->setParameter("client", client_));

  for (const auto& segment_connection : owner_->segment_connections_plan_) {
    for (auto connection : segment_connection.port_maps) {
      /**
       * Key interface how GraphDriver acquires C++ API Segment Connection Map
      */
      graph_driver_->addSegmentConnection(connection.tx.to_string(),
        connection.rx.to_string());
      GXF_LOG_INFO("set connection plan to driver: %s -> %s",
        connection.tx.to_string().c_str(), connection.rx.to_string().c_str());
    }
  }

  return Success;
}

}  // namespace gxf
}  // namespace nvidia
