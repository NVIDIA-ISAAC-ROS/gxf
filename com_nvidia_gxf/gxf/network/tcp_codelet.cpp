/*
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/network/tcp_codelet.hpp"

#include <functional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/std/gems/utils/time.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t TcpCodelet::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(receivers_, "receivers", "Entity receivers",
                                 "List of receivers to receive entities from", {});
  result &= registrar->parameter(transmitters_, "transmitters", "Entity transmitters",
                                 "List of transmitters to publish entities to", {});
  result &= registrar->parameter(entity_serializer_, "entity_serializer", "Entity serializer",
                                 "Serializer for serializing entities");
  result &= registrar->parameter(address_, "address", "Address", "Address for TCP connection");
  result &= registrar->parameter(port_, "port", "Port", "Port for TCP connection");
  result &= registrar->parameter(
      timeout_ms_, "timeout_ms", "Connection timeout",
      "Time in milliseconds to wait before retrying connection. Deprecated - use timeout_period "
      "instead.",
      Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      timeout_text_, "timeout_period", "Connection timeout",
      "Time to wait before retrying connection. The period is specified as a string containing a "
      "number and an (optional) unit. If no unit is given the value is assumed to be in "
      "nanoseconds. Supported units are: Hz, s, ms. Example: 10ms, 10000000, 0.2s, 50Hz",
      std::string("100ms"));
  result &=
      registrar->parameter(maximum_attempts_, "maximum_attempts", "Maximum attempts",
                           "Maximum number of attempts for I/O operations before failing", 10UL);
  result &= registrar->parameter(
      async_scheduling_term_, "async_scheduling_term", "Asynchronous Scheduling Term",
      "Schedules execution when TCP socket or receivers have a message.");
  result &= registrar->parameter(
      max_msg_delay_ms_, "max_msg_delay_ms", "Max message delay [ms]",
      "Time in milliseconds to wait between messages before ending connection. "
      "Helpful for debugging.",
      Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      max_duration_ms_, "max_duration_ms", "Max duration [ms]",
      "The maximum duration for which the component will run (in ms). If not specified the "
      "component will run indefinitely, unless another termination condition is specified. "
      "Helpful for debugging.",
      Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      max_connection_attempts_, "max_connection_attempts", "Max connection attempts",
      "The maximum number of times the component will attempt to reconnect. If not specified the "
      "component will attempt reconnection indefinitely, unless another termination condition is "
      "specified. Helpful for debugging.",
      Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  return ToResultCode(result);
}

gxf_result_t TcpCodelet::initialize() {
  const auto maybe_timeout = ParseRecessPeriodString(timeout_text_, cid());
  if (!maybe_timeout) { return maybe_timeout.error(); }
  timeout_ns_ = maybe_timeout.value();
  if (timeout_ms_.try_get()) {
    GXF_LOG_WARNING(
        "'timeout_ms' parameter in TcpClient and TcpServer is deprecated. Use 'timeout_period' "
        "instead. Overriding timeout_period value.");
    timeout_ns_ = timeout_ms_.try_get().value() * 1000UL;
  }
  // Initialize channel map
  channel_map_.clear();
  for (Handle<Transmitter> transmitter : transmitters_.get()) {
    const uint64_t channel_id = std::hash<std::string>{}(transmitter->name());
    if (channel_map_.count(channel_id)) {
      GXF_LOG_ERROR("Expecting unique transmitter names, encountered duplicate: %s.",
                    transmitter->name());
      return GXF_PARAMETER_ALREADY_REGISTERED;
    }
    channel_map_[channel_id] = transmitter;
  }

  // Open sockets
  auto result = this->openSockets();
  if (!result) { return ToResultCode(result); }

  // Start monitor thread
  monitor_future_ = std::async(std::launch::async, [this] { return monitor(); });

  return GXF_SUCCESS;
}

gxf_result_t TcpCodelet::deinitialize() {
  return ToResultCode(this->closeSockets());
}

gxf_result_t TcpCodelet::tick() {
  if (monitor_future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    async_scheduling_term_->setEventState(gxf::AsynchronousEventState::EVENT_NEVER);
    return GXF_SUCCESS;
  }

  if (client_socket_.available()) {
    auto result = client_socket_.receiveMessages(context(), entity_serializer_.get());
    if (result) {
      tx_messages_ = result.value();
      last_msg_timestamp_ = std::chrono::steady_clock::now();
    } else {
      GXF_LOG_WARNING(
          "Encountered error %s while deserializing message(s). Skipping message "
          "deserialization.",
          GxfResultStr(result.error()));
    }
  }

  // Send messages from receivers through socket
  for (Handle<Receiver> receiver : available_receivers_) {
    Expected<Entity> entity = receiver->receive();
    if (!entity) {
      // Receiver has no message, so move to the next receiver
      GXF_LOG_WARNING(
          "Receiver %s had data available before tick, but has no message on receive. Skipping.",
          receiver->name());
      continue;
    }
    MessageHeader header;
    header.channel_id = std::hash<std::string>{}(receiver->name());
    rx_messages_.push_back({header, std::move(entity.value())});
  }
  if (!rx_messages_.empty()) {
    Expected<size_t> size = client_socket_.sendMessages(rx_messages_, entity_serializer_.get());
    if (!size) {
      GXF_LOG_WARNING("Error when attempting to send message: %s", size.get_error_message());
      // Failing to send messages is not a fatal error since connection to the client
      // may have been broken. Forwarding the error will terminate the server and stop the graph.
      // TODO(ayusmans): new error codes to distinguish non-fatal errors
    }
    rx_messages_.clear();
  }
  available_receivers_.clear();

  // Send messages from socket through transmitters
  for (const TcpMessage& message : tx_messages_) {
    // Find message channel and publish message
    const auto iterator = channel_map_.find(message.header.channel_id);
    if (iterator == channel_map_.end()) {
      // Save unknown channel in map
      GXF_LOG_WARNING("Channel ID 0x%zx not found", message.header.channel_id);
      channel_map_[message.header.channel_id] = Handle<Transmitter>::Null();
      continue;
    }
    Handle<Transmitter> transmitter = iterator->second;
    if (!transmitter) {
      GXF_LOG_WARNING("Transmitter %s not found, skipping message.", transmitter->name());
      continue;
    }
    Expected<void> result = transmitter->publish(message.entity);
    if (!result) { return ToResultCode(result); }
  }
  tx_messages_.clear();

  // Reset event state so monitor thread can resume polling socket/receivers
  async_scheduling_term_->setEventState(gxf::AsynchronousEventState::EVENT_WAITING);

  return GXF_SUCCESS;
}

gxf_result_t TcpCodelet::stop() {
  async_scheduling_term_->setEventState(gxf::AsynchronousEventState::EVENT_NEVER);
  return ToResultCode(monitor_future_.get());
}

Expected<void> TcpCodelet::monitor() {
  monitor_start_timestamp_ = std::chrono::steady_clock::now();
  uint64_t connection_attempts = 0;
  while (true) {
    if (async_scheduling_term_->getEventState() == AsynchronousEventState::READY) {
      async_scheduling_term_->setEventState(gxf::AsynchronousEventState::EVENT_WAITING);
    }
    if (async_scheduling_term_->getEventState() == AsynchronousEventState::EVENT_WAITING) {
      if (client_socket_.connected()) {
        // Reset connection attempts
        connection_attempts = 0;
        // Schedule entity if socket has data available
        if (client_socket_.available()) {
          async_scheduling_term_->setEventState(AsynchronousEventState::EVENT_DONE);
        }
        // Schedule entity if any receivers have data available
        for (Handle<Receiver> receiver : receivers_.get()) {
          if (receiver->back_size() + receiver->size() > 0) {
            available_receivers_.push_back(receiver);
            last_msg_timestamp_ = std::chrono::steady_clock::now();
          }
        }
        if (available_receivers_.size()) {
          async_scheduling_term_->setEventState(AsynchronousEventState::EVENT_DONE);
        }
      } else {
        if (max_connection_attempts_.try_get()) {
          auto max_connection_attempts = max_connection_attempts_.try_get().value();
          if (connection_attempts == max_connection_attempts) {
            GXF_LOG_ERROR("Reached maximum number of connection attempts.");
            async_scheduling_term_->setEventState(AsynchronousEventState::EVENT_DONE);
            return Unexpected{GXF_CONNECTION_ATTEMPTS_EXCEEDED};
          }
          GXF_LOG_WARNING("Trying to reconnect, attempt %lu of %lu.", ++connection_attempts,
                          max_connection_attempts);
        }
        std::this_thread::sleep_for(std::chrono::nanoseconds(timeout_ns_));
        this->reconnectSockets();
      }
      // Test message delay condition to see if TCP connection should end
      if (last_msg_timestamp_.has_value() && max_msg_delay_ms_.try_get()) {
        const auto now = std::chrono::steady_clock::now();
        const auto delta =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_msg_timestamp_.value())
                .count();
        const auto max_msg_delay_ms = max_msg_delay_ms_.try_get().value();
        if (delta > max_msg_delay_ms) {
          GXF_LOG_WARNING("Time between messages has exceeded %ld ms, stopping %s.",
                          max_msg_delay_ms, name());
          async_scheduling_term_->setEventState(AsynchronousEventState::EVENT_DONE);
          break;
        }
      }
      // Test max duration condition to see if TCP connection should end
      if (max_duration_ms_.try_get()) {
        const auto now = std::chrono::steady_clock::now();
        const auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(
                               now - monitor_start_timestamp_.value())
                               .count();
        const auto max_duration_ms = max_duration_ms_.try_get().value();
        if (delta > max_duration_ms) {
          GXF_LOG_WARNING("Execution time has exceeded max duration (%ld ms), stopping %s.",
                          max_duration_ms, name());
          async_scheduling_term_->setEventState(AsynchronousEventState::EVENT_DONE);
          break;
        }
      }
    } else if (async_scheduling_term_->getEventState() == AsynchronousEventState::EVENT_NEVER) {
      break;
    }
  }
  return Success;
}

}  // namespace gxf
}  // namespace nvidia
