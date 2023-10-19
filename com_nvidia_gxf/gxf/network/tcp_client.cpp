/*
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/network/tcp_client.hpp"

namespace nvidia {
namespace gxf {

Expected<void> TcpClient::openSockets() {
  return client_socket_.open();
}

Expected<void> TcpClient::reconnectSockets() {
  // note - referencing https://stackoverflow.com/a/16214433
  client_socket_.close();
  client_socket_.open();
  client_socket_.connect(address_.get().c_str(), port_);
  return Success;
}

Expected<void> TcpClient::closeSockets() {
  return client_socket_.close();
}

}  // namespace gxf
}  // namespace nvidia
