/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_NETWORK_TCP_SERVER_SOCKET_HPP_
#define NVIDIA_GXF_NETWORK_TCP_SERVER_SOCKET_HPP_

#include <string>

#include "gxf/network/tcp_client_socket.hpp"

namespace nvidia {
namespace gxf {

// Object for managing a TCP socket on the server side.
// Connection must be established prior to use.
// Uses a TcpClientSocket as a proxy for communication.
class TcpServerSocket {
 public:
  TcpServerSocket(const char* address, uint16_t port)
    : address_{address}, port_{port}, socket_{-1} {}
  TcpServerSocket() : address_{"0.0.0.0"}, port_{0}, socket_{-1} {}
  ~TcpServerSocket() = default;
  TcpServerSocket(const TcpServerSocket& other) = delete;
  TcpServerSocket(TcpServerSocket&& other) = default;
  TcpServerSocket& operator=(const TcpServerSocket& other) = delete;
  TcpServerSocket& operator=(TcpServerSocket&& other) = default;

  // Initializes server socket.
  // Creates socket file descriptor.
  // Binds IP address to socket.
  // Listens for incoming connections on socket.
  Expected<void> open();
  // Deinitializes server socket.
  // Destroys socket file descriptor.
  Expected<void> close();
  // Attempts to connect to a TCP client.
  // Returns a TCP client socket that can be used as an endpoint.
  // Call will fail if there are no clients requesting to connect.
  Expected<TcpClientSocket> connect();

 private:
  // Server address
  std::string address_;
  // Server port
  uint16_t port_;
  // Socket file descriptor
  int socket_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_NETWORK_TCP_SERVER_SOCKET_HPP_
