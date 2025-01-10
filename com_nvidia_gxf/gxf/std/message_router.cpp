/*
Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/message_router.hpp"

#include <map>
#include <set>
#include <string>
#include <utility>

#include "gxf/core/expected_macro.hpp"
#include "gxf/std/connection.hpp"
#include "gxf/std/timestamp.hpp"
#include "gxf/std/topic.hpp"

namespace nvidia {
namespace gxf {

Expected<void> MessageRouter::addRoutes(const Entity& entity) {
  // Add routes created explicitly with `gxf::Connection`.
  const auto connections = UNWRAP_OR_RETURN(entity.findAllHeap<Connection>());
  for (auto maybe_connection : connections) {
    Handle<Connection> connection = UNWRAP_OR_RETURN(maybe_connection);
    RETURN_IF_ERROR(connect(connection->source(), connection->target()));
  }

  // Add routes created implicitly with `gxf::Topic`.
  const auto topics = UNWRAP_OR_RETURN(entity.findAllHeap<Topic>());
  for (auto maybe_topic : topics) {
    Handle<Topic> topic = UNWRAP_OR_RETURN(maybe_topic);
    const std::string topic_name = topic->getTopicName();
    for (Handle<Transmitter> transmitter : topic->getTransmitters()) {
      RETURN_IF_ERROR(registerTransmitter(topic_name, transmitter));
    }
    for (Handle<Receiver> receiver : topic->getReceivers()) {
      RETURN_IF_ERROR(registerReceiver(topic_name, receiver));
    }
  }

  //  caching receivers with message router
  const auto receivers = UNWRAP_OR_RETURN(entity.findAllHeap<Receiver>());
  for (auto maybe_rx : receivers) {
    const Handle<Receiver> rx = UNWRAP_OR_RETURN(maybe_rx);
    receivers_[entity.eid()].insert(rx);
  }

  //  caching transmitters with message router
  const auto transmitters = UNWRAP_OR_RETURN(entity.findAllHeap<Transmitter>());
  for (auto maybe_tx : transmitters) {
    const Handle<Transmitter> tx = UNWRAP_OR_RETURN(maybe_tx);
    transmitters_[entity.eid()].insert(tx);
  }
  return Success;
}

Expected<void> MessageRouter::removeRoutes(const Entity& entity) {
  // Remove routes created explicitly with `gxf::Connection`.
  const auto connections = UNWRAP_OR_RETURN(entity.findAllHeap<Connection>());
  for (auto maybe_connection : connections) {
    Handle<Connection> connection = UNWRAP_OR_RETURN(maybe_connection);
    RETURN_IF_ERROR(disconnect(connection->source(), connection->target()));
  }

  // Remove routes created implicitly with `gxf::Topic`.
  const auto topics = UNWRAP_OR_RETURN(entity.findAllHeap<Topic>());
  for (auto maybe_topic : topics) {
    Handle<Topic> topic = UNWRAP_OR_RETURN(maybe_topic);
    const std::string topic_name = topic->getTopicName();
    for (Handle<Transmitter> transmitter : topic->getTransmitters()) {
      RETURN_IF_ERROR(deregisterTransmitter(topic_name, transmitter));
    }
    for (Handle<Receiver> receiver : topic->getReceivers()) {
      RETURN_IF_ERROR(deregisterReceiver(topic_name, receiver));
    }
  }
  return Success;
}

Expected<void> MessageRouter::connect(Handle<Transmitter> tx, Handle<Receiver> rx) {
  if (!tx || !rx) { return Unexpected{GXF_ARGUMENT_NULL}; }
  GXF_LOG_DEBUG("Registering a connection from '%s' to '%s'.", tx.name(), rx.name());
  routes_[tx].insert(rx);
  routes_reversed_[rx].insert(tx);
  rx->setTransmitter(tx);
  return Success;
}

Expected<void> MessageRouter::disconnect(Handle<Transmitter> tx, Handle<Receiver> rx) {
  if (!tx || !rx) { return Unexpected{GXF_ARGUMENT_NULL}; }
  GXF_LOG_DEBUG("Deregistering a connection from '%s' to '%s'.", tx.name(), rx.name());

  // Find all receivers corresponding to the transmitter.
  const auto tx_and_receivers = routes_.find(tx);
  if (tx_and_receivers == routes_.end()) {
    return Unexpected{GXF_ENTITY_COMPONENT_NOT_FOUND};
  }
  auto& receivers = tx_and_receivers->second;

  // Find the specific receiver and erase it.
  const auto rx_it = receivers.find(rx);
  if (rx_it == receivers.end()) { return Unexpected{GXF_ENTITY_COMPONENT_NOT_FOUND}; }
  receivers.erase(rx_it);

  const auto transmitters_and_rx = routes_reversed_.find(rx);
  if (transmitters_and_rx == routes_reversed_.end()) {
    return Unexpected{GXF_ENTITY_COMPONENT_NOT_FOUND};
  }
  auto& transmitters = transmitters_and_rx->second;

  // Find the specific transmitter and erase it
  const auto tx_it = transmitters.find(tx);
  if (tx_it == transmitters.end()) { return Unexpected{GXF_ENTITY_COMPONENT_NOT_FOUND}; }
  transmitters.erase(tx_it);

  return Success;
}

Expected<void> MessageRouter::registerTransmitter(const std::string& topic_name,
                                                  Handle<Transmitter> tx) {
  if (!tx) {
    GXF_LOG_ERROR("Received null handle for topic '%s'.", topic_name.c_str());
    return Unexpected{GXF_ARGUMENT_NULL};
  }
  GXF_LOG_INFO("Registering transmitter '%s' for topic '%s'.", tx.name(), topic_name.c_str());
  topic_and_transmitters_[topic_name].insert(tx);
  transmitter_and_topic_[tx] = topic_name;

  return Success;
}

Expected<void> MessageRouter::deregisterTransmitter(const std::string& topic_name,
                                                    Handle<Transmitter> tx) {
  if (!tx) {
    GXF_LOG_ERROR("Received null handle for topic '%s'.", topic_name.c_str());
    return Unexpected{GXF_ARGUMENT_NULL};
  }
  GXF_LOG_INFO("Deregistering transmitter '%s' for topic '%s'.", tx.name(), topic_name.c_str());
  topic_and_transmitters_[topic_name].erase(tx);
  transmitter_and_topic_.erase(tx);

  return Success;
}

Expected<void> MessageRouter::registerReceiver(const std::string& topic_name, Handle<Receiver> rx) {
  if (!rx) {
    GXF_LOG_ERROR("Received null handle for topic '%s'.", topic_name.c_str());
    return Unexpected{GXF_ARGUMENT_NULL};
  }
  GXF_LOG_INFO("Registering receiver '%s' for topic '%s'.", rx.name(), topic_name.c_str());
  topic_and_receivers_[topic_name].insert(rx);
  receiver_and_topic_[rx] = topic_name;
  return Success;
}

Expected<void> MessageRouter::deregisterReceiver(const std::string& topic_name,
                                                 Handle<Receiver> rx) {
  if (!rx) {
    GXF_LOG_ERROR("Received null handle for topic '%s'.", topic_name.c_str());
    return Unexpected{GXF_ARGUMENT_NULL};
  }
  GXF_LOG_INFO("Deregistering receiver '%s' for topic '%s'.", rx.name(), topic_name.c_str());
  topic_and_receivers_[topic_name].erase(rx);
  receiver_and_topic_.erase(rx);
  return Success;
}

Expected<Handle<Receiver>> MessageRouter::getRx(Handle<Transmitter> tx) {
  const auto receivers = UNWRAP_OR_RETURN(getConnectedReceivers(tx));
  if (receivers.empty()) { return Unexpected{GXF_ARGUMENT_NULL}; }
  if (receivers.size() > 1) { return Unexpected{GXF_ARGUMENT_INVALID}; }
  return *receivers.begin();
}

Expected<std::set<Handle<Receiver>>> MessageRouter::getConnectedReceivers(
    Handle<Transmitter> tx) const {
  if (!tx) {
    GXF_LOG_ERROR("Received null handle.");
    return Unexpected{GXF_ARGUMENT_NULL};
  }

  std::set<Handle<Receiver>> receivers;

  // Add receivers from connections.
  const auto connection_receivers = routes_.find(tx);
  if (connection_receivers != routes_.end()) {
    receivers.insert(connection_receivers->second.begin(), connection_receivers->second.end());
  }

  // Add receivers from topics.
  const auto topic_name = transmitter_and_topic_.find(tx);
  if (topic_name != transmitter_and_topic_.end()) {
    const auto topic_receivers = topic_and_receivers_.find(topic_name->second);
    if (topic_receivers != topic_and_receivers_.end()) {
      receivers.insert(topic_receivers->second.begin(), topic_receivers->second.end());
    }
  }

  return receivers;
}

Expected<std::set<Handle<Transmitter>>> MessageRouter::getConnectedTransmitters(
    Handle<Receiver> rx) const {
  if (!rx) {
    GXF_LOG_ERROR("Received null handle.");
    return Unexpected{GXF_ARGUMENT_NULL};
  }

  std::set<Handle<Transmitter>> transmitters;

  // Add transmitters from connections.
  const auto connection_transmitters = routes_reversed_.find(rx);
  if (connection_transmitters != routes_reversed_.end()) {
    transmitters.insert(connection_transmitters->second.begin(),
                        connection_transmitters->second.end());
  }

  // Add transmitters from topics.
  const auto topic_name = receiver_and_topic_.find(rx);
  if (topic_name != receiver_and_topic_.end()) {
    const auto topic_transmitters = topic_and_transmitters_.find(topic_name->second);
    if (topic_transmitters != topic_and_transmitters_.end()) {
      transmitters.insert(topic_transmitters->second.begin(), topic_transmitters->second.end());
    }
  }

  return transmitters;
}

Expected<void> MessageRouter::syncInbox(const Entity& entity) {
  if (receivers_.find(entity.eid()) != receivers_.end()) {
    const auto& cached_receivers = receivers_[entity.eid()];
    for (auto& rx : cached_receivers) {
      if (!rx) {
        GXF_LOG_ERROR("Invalid Receiver obtained from cached receivers for entity %s",
         entity.name());
        return Unexpected(GXF_FAILURE);
      }

      const auto result = rx->sync();
      if (!result) {
        GXF_LOG_ERROR("Failed to sync receiver %s for entity %s", rx->name(), entity.name());
        return ForwardError(result);
      }
    }
  }

  return Success;
}

Expected<void> MessageRouter::syncOutbox(const Entity& entity) {
  // Sync all transmitters.
  if (transmitters_.find(entity.eid()) != transmitters_.end()) {
    const auto& cached_transmitters = transmitters_[entity.eid()];
    for (auto& tx : cached_transmitters) {
      if (!tx) {
        GXF_LOG_ERROR("Obtained invalid transmitter for entity %s", entity.name());
        return Unexpected(GXF_FAILURE);
      }

      // Sync all transmitters.
      RETURN_IF_ERROR(tx->sync());

      // Distribute the message to any connected receivers.
      const bool has_new_message = tx->size() > 0;
      if (has_new_message) {
        const auto receivers = UNWRAP_OR_RETURN(getConnectedReceivers(tx));
        while (tx->size() > 0) {
          Entity message = UNWRAP_OR_RETURN(tx->pop());

          const auto result = distribute(tx, message, receivers);
          if (!result) {
            GXF_LOG_ERROR("Error while distribution of message from tx %s", tx->name());
            return ForwardError(result);
          }
        }

        // send event to trigger execution of downstream entities.
        for (const Handle<Receiver>& receiver : receivers) {
          GXF_LOG_VERBOSE("Notifying downstream receiver with eid '%ld'.", receiver->eid());
          GxfEntityNotifyEventType(entity.context(), receiver->eid(), GXF_EVENT_MESSAGE_SYNC);
        }
      }
    }
  }

  return Success;
}

Expected<void> MessageRouter::distribute(Handle<Transmitter> tx, const Entity& message,
                                         const std::set<Handle<Receiver>>& receivers) {
  for (const Handle<Receiver>& receiver : receivers) { receiver->push(message); }
  return Success;
}

Expected<void> MessageRouter::setClock(Handle<Clock> clock) {
  return Success;
}

Expected<void> MessageRouter::addNetworkContext(Handle<NetworkContext> context) {
  return Success;
}

}  // namespace gxf
}  // namespace nvidia
