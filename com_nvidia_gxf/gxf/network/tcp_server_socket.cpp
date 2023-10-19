/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/network/tcp_server_socket.hpp"

#include <arpa/inet.h>
#if defined (_QNX_SOURCE)
#include <fcntl.h>
#endif
#include <sys/socket.h>
#include <unistd.h>

namespace nvidia {
namespace gxf {

namespace {

// Maximum number of clients the server can support
constexpr size_t kMaxClients = 1;

}  // namespace

Expected<void> TcpServerSocket::open() {
  int result;
  // Open socket file descriptor
#if defined (_QNX_SOURCE)
  socket_ = ::socket(AF_INET, SOCK_STREAM, 0);
  int oflags = fcntl(socket_, F_GETFL);
  oflags |= O_NONBLOCK;
  fcntl(socket_, oflags);
#else
  socket_ = ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
#endif
  if (socket_ < 0) {
    return Unexpected{GXF_FAILURE};
  }

  // Enable address and port reuse
#if !defined (_QNX_SOURCE)
  int option_value = 1;
  result = setsockopt(socket_, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
                          &option_value, sizeof(option_value));
  if (result != 0) {
    return Unexpected{GXF_FAILURE};
  }
#endif
  // Format IP address
  sockaddr_in ip_address;
  ip_address.sin_family = AF_INET;
  ip_address.sin_port = htons(port_);
  result = inet_pton(ip_address.sin_family, address_.c_str(), &ip_address.sin_addr);
  if (result != 1) {
    GXF_LOG_ERROR("Invalid IP address %s:%u", address_.c_str(), port_);
    return Unexpected{GXF_FAILURE};
  }

  // Bind IP address to socket
  result = ::bind(socket_, reinterpret_cast<sockaddr*>(&ip_address), sizeof(ip_address));
  if (result != 0) {
    return Unexpected{GXF_FAILURE};
  }

  // Listen for incoming connections
  result = ::listen(socket_, kMaxClients);
  if (result != 0) {
    return Unexpected{GXF_FAILURE};
  }

  return Success;
}

Expected<void> TcpServerSocket::close() {
  // Close socket file descriptor
  const int result = ::close(socket_);
  if (result != 0) {
    return Unexpected{GXF_FAILURE};
  }

  return Success;
}

Expected<TcpClientSocket> TcpServerSocket::connect() {
  // Connect to TCP client
  sockaddr_in ip_address;
  socklen_t length = sizeof(ip_address);
  const int client_socket = ::accept(socket_, reinterpret_cast<sockaddr*>(&ip_address), &length);
  if (client_socket < 0) {
    GXF_LOG_WARNING("Failed to connect to TCP client");
    return Unexpected{GXF_FAILURE};
  }

  // Initialize TCP client
  TcpClientSocket client;
  const Expected<void> result = client.openConnectedSocket(client_socket);
  if (!result) {
    return ForwardError(result);
  }

  char address[INET_ADDRSTRLEN];
  if (!inet_ntop(ip_address.sin_family, &ip_address.sin_addr, address, sizeof(address))) {
    return Unexpected{GXF_FAILURE};
  }
  const uint16_t port = ntohs(ip_address.sin_port);
  GXF_LOG_DEBUG("Successfully connected to TCP client %s:%u", address, port);

  return client;
}

}  // namespace gxf
}  // namespace nvidia
