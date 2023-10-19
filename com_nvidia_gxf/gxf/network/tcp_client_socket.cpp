/*
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/network/tcp_client_socket.hpp"

#include <arpa/inet.h>
#if defined (_QNX_SOURCE)
#include <net/netbyte.h>
#else
#include <endian.h>
#endif
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <utility>
#include <vector>

#include "common/memory_utils.hpp"

namespace nvidia {
namespace gxf {

namespace {


// Time in milliseconds to wait for a poll event
constexpr int kTimeout = 0;

}  // namespace

Expected<void> TcpClientSocket::open() {
  // Open socket if connection is not yet established
  if (!connected_) {
    fd_socket_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd_socket_ < 0) {
      return Unexpected{GXF_FAILURE};
    }
  }

  return Success;
}

Expected<void> TcpClientSocket::close() {
  // Close socket file descriptor
  GXF_LOG_INFO("TCP close %u", fd_socket_);
  int result = ::close(fd_socket_);
  if (result != 0) {
    GXF_LOG_ERROR("TCP close error %u", result);
    return Unexpected{GXF_FAILURE};
  }
  connected_ = false;

  return Success;
}

gxf_result_t TcpClientSocket::write_abi(const void* data, size_t size, size_t* bytes_written) {
  if (data == nullptr || bytes_written == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  if (!connected_) {
    return GXF_FAILURE;
  }
  size_t offset = 0;
  for (size_t i = 0; i < maximum_attempts_; i++) {
    const ssize_t bytes = ::send(
        fd_socket_, BytePointer(data) + offset, size - offset, MSG_NOSIGNAL);
    if (bytes == -1) {
      GXF_LOG_ERROR("%s", strerror(errno));
      return GXF_FAILURE;
    }
    offset += static_cast<size_t>(bytes);
    if (offset == size) {
      *bytes_written = offset;
      return GXF_SUCCESS;
    }
  }
  GXF_LOG_WARNING("Maximum number of attempts reached (%zu)", maximum_attempts_);
  GXF_LOG_DEBUG("Sent %zu/%zu bytes", offset, size);
  *bytes_written = offset;
  return GXF_FAILURE;
}

gxf_result_t TcpClientSocket::read_abi(void* data, size_t size, size_t* bytes_read) {
  if (data == nullptr || bytes_read == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  if (!connected_) {
    return GXF_FAILURE;
  }
  if (size == 0) {
    return GXF_SUCCESS;
  }
  size_t offset = 0;
  for (size_t i = 0; i < maximum_attempts_; i++) {
    const ssize_t bytes = ::recv(
        fd_socket_, BytePointer(data) + offset, size - offset, MSG_WAITALL);
    if (bytes == -1) {
      GXF_LOG_ERROR("%s", strerror(errno));
      return GXF_FAILURE;
    } else if (bytes == 0) {
      GXF_LOG_DEBUG("Connection broken");
      connected_ = false;
      return GXF_CONNECTION_BROKEN;
    }
    offset += static_cast<size_t>(bytes);
    if (offset == size) {
      *bytes_read = offset;
      return GXF_SUCCESS;
    }
  }
  GXF_LOG_WARNING("Maximum number of attempts reached (%zu)", maximum_attempts_);
  GXF_LOG_DEBUG("Received %zu/%zu bytes", offset, size);
  *bytes_read = offset;
  return GXF_FAILURE;
}

Expected<void> TcpClientSocket::openConnectedSocket(int socket) {
  connected_ = true;
  fd_socket_ = socket;

  return open();
}

Expected<void> TcpClientSocket::reopenSocket() {
  Expected<void> result = close();
  if (!result) {
    return ForwardError(result);
  }
  result = open();
  if (!result) {
    return ForwardError(result);
  }
  return Success;
}

bool TcpClientSocket::available() const {
  struct pollfd fds;
  int retval = 0;

  fds.fd = fd_socket_;
  fds.events = POLLRDNORM;

  // Create poll
  retval = poll(&fds, 1, kTimeout);
  if (retval < 0) {
    return false;
  }

  if ( (retval == 1) && (fds.revents & POLLRDNORM) == POLLRDNORM ) {
    return true;
  } else {
    return false;
  }
}

Expected<void> TcpClientSocket::connect(const char* address, uint16_t port) {
  // Format IP address
  sockaddr_in ip_address;
  ip_address.sin_family = AF_INET;
  ip_address.sin_port = htons(port);
  int result = inet_pton(ip_address.sin_family, address, &ip_address.sin_addr);
  if (result != 1) {
    GXF_LOG_ERROR("Invalid IP address %s:%u", address, port);
    return Unexpected{GXF_FAILURE};
  }

  // Connect to TCP server
  result = ::connect(fd_socket_, reinterpret_cast<sockaddr*>(&ip_address), sizeof(ip_address));

  if (result != 0) {
    GXF_LOG_WARNING("Failed to connect to TCP server %s:%u", address, port);
    return Unexpected{GXF_FAILURE};
  }
  connected_ = true;

  GXF_LOG_DEBUG("Successfully connected to TCP server %s:%u", address, port);
  return Success;
}

Expected<size_t> TcpClientSocket::sendMessages(const std::vector<TcpMessage>& messages,
                                               EntitySerializer* serializer) {
  if (!serializer) {
    return Unexpected{GXF_ARGUMENT_NULL};
  }

  size_t payload_size = 0;

  // Send TCP header
  TcpHeader tcp_header;
  tcp_header.payload_size = 0;  // FIXME: How can we compute this before serializing?
  tcp_header.entity_count = messages.size();
  Expected<size_t> maybe_size = sendTcpHeader(tcp_header);
  if (!maybe_size) {
    return ForwardError(maybe_size);
  }
  payload_size += maybe_size.value();

  for (const TcpMessage& message : messages) {
    // Send GXF message
    maybe_size = sendMessage(message, serializer);
    if (!maybe_size) {
      return ForwardError(maybe_size);
    }
    payload_size += maybe_size.value();
  }

  return payload_size;
}

Expected<std::vector<TcpMessage>> TcpClientSocket::receiveMessages(gxf_context_t context,
                                                                   EntitySerializer* serializer) {
  if (!serializer) {
    return Unexpected{GXF_ARGUMENT_NULL};
  }

  // Receive TCP header
  const Expected<TcpHeader> maybe_tcp_header = receiveTcpHeader();
  if (!maybe_tcp_header) {
    return ForwardError(maybe_tcp_header);
  }

  std::vector<TcpMessage> messages;
  for (size_t i = 0; i < maybe_tcp_header->entity_count; i++) {
    // Receive GXF message
    Expected<TcpMessage> maybe_message = receiveMessage(context, serializer);
    if (!maybe_message) {
      return ForwardError(maybe_message);
    }
    messages.push_back(maybe_message.value());
  }

  return messages;
}

Expected<size_t> TcpClientSocket::sendMessage(const TcpMessage& message,
                                               EntitySerializer* serializer) {
  if (!serializer) {
    return Unexpected{GXF_ARGUMENT_NULL};
  }
  const Expected<size_t> maybe_header_size = sendMessageHeader(message.header);
  if (!maybe_header_size) {
    return ForwardError(maybe_header_size);
  }
  const Expected<size_t> maybe_entity_size = serializer->serializeEntity(message.entity, this);
  if (!maybe_entity_size) {
    return ForwardError(maybe_entity_size);
  }
  return maybe_header_size.value() + maybe_entity_size.value();
}

Expected<TcpMessage> TcpClientSocket::receiveMessage(gxf_context_t context,
                                                     EntitySerializer* serializer) {
  if (!serializer) {
    return Unexpected{GXF_ARGUMENT_NULL};
  }
  const Expected<MessageHeader> maybe_message_header = receiveMessageHeader();
  if (!maybe_message_header) {
    return ForwardError(maybe_message_header);
  }
  const Expected<Entity> maybe_entity = serializer->deserializeEntity(context, this);
  if (!maybe_entity) {
    return ForwardError(maybe_entity);
  }
  return TcpMessage{maybe_message_header.value(), std::move(maybe_entity.value())};
}

Expected<size_t> TcpClientSocket::sendTcpHeader(TcpHeader header) {
  header.payload_size = htole64(header.payload_size);
  header.entity_count = htole64(header.entity_count);
  const Expected<size_t> result = writeTrivialType(&header);
  if (!result) {
    return ForwardError(result);
  }
  return sizeof(header);
}

Expected<TcpHeader> TcpClientSocket::receiveTcpHeader() {
  TcpHeader header;
  const Expected<size_t> result = readTrivialType(&header);
  if (!result) {
    return ForwardError(result);
  }
  header.payload_size = le64toh(header.payload_size);
  header.entity_count = le64toh(header.entity_count);
  return header;
}

Expected<size_t> TcpClientSocket::sendMessageHeader(MessageHeader header) {
  header.channel_id = htole64(header.channel_id);
  const Expected<size_t> result = writeTrivialType(&header);
  if (!result) {
    return ForwardError(result);
  }
  return sizeof(header);
}

Expected<MessageHeader> TcpClientSocket::receiveMessageHeader() {
  MessageHeader header;
  const Expected<size_t> result = readTrivialType(&header);
  if (!result) {
    return ForwardError(result);
  }
  header.channel_id = le64toh(header.channel_id);
  return header;
}

}  // namespace gxf
}  // namespace nvidia
