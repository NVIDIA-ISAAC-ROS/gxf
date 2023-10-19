/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gxf/network/tcp_server.hpp"

#include <utility>

namespace nvidia {
namespace gxf {

Expected<void> TcpServer::openSockets() {
  server_socket_ = TcpServerSocket(address_.get().c_str(), static_cast<uint16_t>(port_));
  return server_socket_.open();
}

Expected<void> TcpServer::reconnectSockets() {
  Expected<TcpClientSocket> client = server_socket_.connect();
  if (client) {
    client_socket_ = std::move(client.value());
    client_socket_.setMaximumAttempts(maximum_attempts_);
  }
  return Success;
}

Expected<void> TcpServer::closeSockets() {
  return client_socket_.close().and_then([&]() { return server_socket_.close(); });
}

}  // namespace gxf
}  // namespace nvidia
