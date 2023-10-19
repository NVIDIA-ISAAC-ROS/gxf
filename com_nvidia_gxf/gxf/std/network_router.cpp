/*
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

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
    return context_->addRoutes(entity);
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
    // const int64_t now = clock_.get()->timestamp();
    const auto transmitters = entity.findAll<Transmitter>();
    if (!transmitters) {
        return ForwardError(transmitters);
    }
    for (auto tx : transmitters.value()) {
        if (!tx) {
        GXF_LOG_ERROR("Found a bad transmitter while syncing outbox for entity %s", entity.name());
        return Unexpected{GXF_FAILURE};
        }
        const auto result = tx.value()->sync_io();
        if (!result) return result;
    }
    return Success;
}

Expected<void> NetworkRouter::syncInbox(const Entity& entity) {
  const auto receivers = entity.findAll<Receiver>();
  if (!receivers) {
    return ForwardError(receivers);
  }
  for (auto rx : receivers.value()) {
    if (!rx) {
      GXF_LOG_ERROR("Found a bad reciever while syncing inbox for entity %s", entity.name());
      return Unexpected{GXF_FAILURE};
    }
    const auto result = rx.value()->sync_io();
    if (!result) return result;
  }
  return Success;
}

Expected<void> NetworkRouter::addNetworkContext(Handle<NetworkContext> context) {
  if (context) {
    context_ = context;
  }
  return Success;
}

}  // namespace gxf
}  // namespace nvidia
