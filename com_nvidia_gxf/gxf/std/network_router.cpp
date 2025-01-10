/*
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <vector>

#include "network_context.hpp"
#include "network_router.hpp"

namespace nvidia {
namespace gxf {

Expected<void> NetworkRouter::removeRoutes(const Entity& entity) {
  if (context_) {
    return context_->removeRoutes(entity);
  }
  return Success;
}

Expected<void> NetworkRouter::addRoutes(const Entity& entity) {
  if (context_) {
    auto maybe = context_->addRoutes(entity);
    if (!maybe) {
      GXF_LOG_ERROR("Failed to do NetworkContext::addRoutes() on entity[eid: %ld, name: %s]."
        " NetworkRouter::addRoutes() early return without caching Tx/Rx",
        entity.eid(), entity.name());
      return maybe;
    }
  }
  //  caching receivers with message router
  const auto receivers = UNWRAP_OR_RETURN(entity.findAllHeap<Receiver>());
  for (auto maybe_rx : receivers) {
    const Handle<Receiver> rx = UNWRAP_OR_RETURN(maybe_rx);
    GXF_LOG_VERBOSE("NetworkRouter caching entity[eid: %ld, name: %s] -> Rx[cid: %ld, name: %s]",
      entity.eid(), entity.name(), rx->cid(), rx->name());
    receivers_[entity.eid()].insert(rx);
  }

  //  caching transmitters with message router
  const auto transmitters = UNWRAP_OR_RETURN(entity.findAllHeap<Transmitter>());
  for (auto maybe_tx : transmitters) {
    const Handle<Transmitter> tx = UNWRAP_OR_RETURN(maybe_tx);
    GXF_LOG_VERBOSE("NetworkRouter caching entity[eid: %ld, name: %s] -> Tx[cid: %ld, name: %s]",
      entity.eid(), entity.name(), tx->cid(), tx->name());
    transmitters_[entity.eid()].insert(tx);
  }
  return Success;
}

Expected<void> NetworkRouter::setClock(Handle<Clock> clock) {
  if (!clock) { return Unexpected{GXF_ARGUMENT_NULL}; }
  clock_ = clock;
  return Success;
}

Expected<void> NetworkRouter::syncOutbox(const Entity& entity) {
    if (!clock_) { return Unexpected{GXF_PARAMETER_NOT_INITIALIZED}; }
    if (transmitters_.find(entity.eid()) != transmitters_.end()) {
      const auto& cached_transmitters = transmitters_.at(entity.eid());
      for (auto& tx : cached_transmitters) {
        if (!tx) {
          GXF_LOG_ERROR("Found a bad transmitter while syncing outbox for entity %s",
           entity.name());
          return Unexpected(GXF_ENTITY_COMPONENT_NOT_FOUND);
        }
        const auto result = tx->sync_io();
        if (!result) return result;
      }
    }
    return Success;
}

Expected<void> NetworkRouter::syncInbox(const Entity& entity) {
  if (receivers_.find(entity.eid()) != receivers_.end()) {
    const auto& cached_receivers = receivers_.at(entity.eid());
    for (auto& rx : cached_receivers) {
      if (!rx) {
        GXF_LOG_ERROR("Found a bad receiver while syncing inbox for entity %s", entity.name());
        return Unexpected{GXF_ENTITY_COMPONENT_NOT_FOUND};
      }
      const auto result = rx->sync_io();
      if (!result) return result;
    }
  }
  return Success;
}

Expected<void> NetworkRouter::wait(const Entity& entity) {
  gxf_result_t result;
  // Get the Expected object wrapping the FixedVector
  const auto expectedReceivers = entity.findAllHeap<Receiver>();
  if (!expectedReceivers) {
      return ExpectedOrCode(GXF_FAILURE);
  }

  // Extract the FixedVector from the Expected object
  const auto& receivers = expectedReceivers.value();
  // Initially, all receivers are "pending"
  std::vector<Handle<Receiver>> pendingReceivers;
  for (const auto& handle : receivers) {
    pendingReceivers.push_back(handle.value());
  }
  while (!pendingReceivers.empty()) {
    std::vector<Handle<Receiver>> stillPending;
    for (const auto& handle : pendingReceivers) {
      // Assuming Handle provides a get() method to retrieve the pointer
      auto* receiver = handle.get();
      if (!receiver) {
        continue;
      }
      result = receiver->wait_abi();
      if (result == GXF_FAILURE) {
        return ExpectedOrCode(GXF_FAILURE);
      }
      if (result == GXF_NOT_FINISHED) {
        stillPending.push_back(handle);
      }
    }
    // Swap lists, so we only iterate over receivers that are still pending in the next round
    pendingReceivers.swap(stillPending);
  }
  return Success;
}

Expected<void> NetworkRouter::addNetworkContext(Handle<NetworkContext> context) {
  if (context) {
    context_ = context;
    auto res = context_->init_context();
    if (res != GXF_SUCCESS) {
      GXF_LOG_ERROR("Network Context init_context failed");
      return Unexpected{GXF_FAILURE};
    }
  }
  return Success;
}

}  // namespace gxf
}  // namespace nvidia
