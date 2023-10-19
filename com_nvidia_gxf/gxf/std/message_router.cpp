/*
Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/connection.hpp"
#include "gxf/std/message_router.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace gxf {

Expected<void> MessageRouter::addRoutes(const Entity& entity) {
  const auto connections = entity.findAll<Connection>();
  if (!connections) {
    return ForwardError(connections);
  }
  for (auto connection : connections.value()) {
    if (!connection) {
      GXF_LOG_ERROR("Found a bad connection while adding routes");
      return Unexpected{GXF_FAILURE};
    }
    const auto result = connect(connection.value()->source(), connection.value()->target());
    if (!result) {
      return result;
    }
  }
  return Success;
}

Expected<void> MessageRouter::removeRoutes(const Entity& entity) {
  const auto connections = entity.findAll<Connection>();
  if (!connections) {
    return ForwardError(connections);
  }
  for (auto connection : connections.value()) {
    if (!connection) {
      GXF_LOG_ERROR("Found a bad connection while removing routes");
      return Unexpected{GXF_FAILURE};
    }
    const auto result = disconnect(connection.value()->source(), connection.value()->target());
    if (!result) {
      return result;
    }
  }
  return Success;
}

Expected<void> MessageRouter::connect(Handle<Transmitter> tx, Handle<Receiver> rx) {
  if (!tx || !rx) {
    return Unexpected{GXF_ARGUMENT_NULL};
  }
  const auto it = routes_.find(tx);
  if (it != routes_.end()) {
    GXF_LOG_ERROR("Transmitter can only be connected once to a single receiver."
                  "Tx %s is already connected to Rx %s", tx->name(), it->second->name());
    return Unexpected{GXF_FAILURE};
  }
  routes_[tx] = rx;
  return Success;
}

Expected<void> MessageRouter::disconnect(Handle<Transmitter> tx, Handle<Receiver> rx) {
  if (!tx) {
    return Unexpected{GXF_ARGUMENT_NULL};
  }
  const auto it = routes_.find(tx);
  if (it == routes_.end()) {
    return Unexpected{GXF_ENTITY_COMPONENT_NOT_FOUND};
  }
  if (rx != it->second) {
    GXF_LOG_ERROR("Tx %s is connected to %s and not %s."
                  " Disconnect operation failed", tx->name(), it->second->name(), rx->name());
    return Unexpected{GXF_FAILURE}; }

  routes_.erase(it);
  return Success;
}

Expected<Handle<Receiver>> MessageRouter::getRx(Handle<Transmitter> tx) {
  if (!tx) {
    return Unexpected{GXF_ARGUMENT_NULL};
  }
  const auto it = routes_.find(tx);
  if (it == routes_.end()) {
    GXF_LOG_ERROR("Connection not found for Tx %s", tx->name());
    return Unexpected{GXF_FAILURE};
  }
  return it->second;
}

Expected<void> MessageRouter::syncInbox(const Entity& entity) {
  const auto receivers = entity.findAll<Receiver>();
  if (!receivers) {
    return ForwardError(receivers);
  }
  for (auto rx : receivers.value()) {
    if (!rx) {
      GXF_LOG_ERROR("Found a bad reciever while syncing inbox for entity %s", entity.name());
      return Unexpected{GXF_FAILURE};
    }
    const auto result = rx.value()->sync();
    if (!result) return result;
  }
  return Success;
}

Expected<void> MessageRouter::syncOutbox(const Entity& entity) {
  if (!clock_) { return Unexpected{GXF_PARAMETER_NOT_INITIALIZED}; }
  const int64_t now = clock_.get()->timestamp();

  const auto transmitters = entity.findAll<Transmitter>();
  if (!transmitters) {
    return ForwardError(transmitters);
  }
  for (auto tx : transmitters.value()) {
    if (!tx) {
      GXF_LOG_ERROR("Found a bad transmitter while syncing outbox for entity %s", entity.name());
      return Unexpected{GXF_FAILURE};
    }
    const auto result = tx.value()->sync();
    if (!result) return result;
  }

  for (auto tx : transmitters.value()) {
    if (!tx) {
      GXF_LOG_ERROR("Found a bad transmitter while syncing outbox for entity %s", entity.name());
      return Unexpected{GXF_FAILURE};
    }
    while (tx.value()->size() > 0) {
      const auto result_1 = tx.value()->pop();
      if (!result_1) return Unexpected{result_1.error()};
      auto timestamp = result_1.value().get<Timestamp>("timestamp");
      if (timestamp) { timestamp.value()->pubtime = now; }
      const auto result_2 = distribute(tx.value(), result_1.value());
      if (!result_2) return result_2;
    }
  }

  return Success;
}

Expected<void> MessageRouter::distribute(Handle<Transmitter> tx, const Entity& message) {
  if (!tx) {
    return Unexpected{GXF_ARGUMENT_NULL};
  }
  const auto it = routes_.find(tx);
  if (it == routes_.end()) {
    // No receiver is connected to this transmitter
    return Success;
  }
  return it->second->push(message);
}

Expected<void> MessageRouter::setClock(Handle<Clock> clock) {
  if (!clock) { return Unexpected{GXF_ARGUMENT_NULL}; }
  clock_ = clock;
  return Success;
}

Expected<void> MessageRouter::addNetworkContext(Handle<NetworkContext> context) {
  return Success;
}

}  // namespace gxf
}  // namespace nvidia
