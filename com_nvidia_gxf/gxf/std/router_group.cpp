/*
Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/router_group.hpp"

#include <utility>

namespace nvidia {
namespace gxf {

Expected<void> RouterGroup::addRoutes(const Entity& entity) {
  Expected<void> result = Success;
  for (size_t i = 0; i < routers_.size(); i++) {
    const auto& router = routers_.at(i).value();
    result &= router->addRoutes(entity);
  }
  return result;
}

Expected<void> RouterGroup::removeRoutes(const Entity& entity) {
  Expected<void> result = Success;
  for (size_t i = 0; i < routers_.size(); i++) {
    const auto& router = routers_.at(i).value();
    result &= router->removeRoutes(entity);
  }
  return result;
}

Expected<void> RouterGroup::syncInbox(const Entity& entity) {
  Expected<void> result = Success;
  for (size_t i = 0; i < routers_.size(); i++) {
    const auto& router = routers_.at(i).value();
    result &= router->syncInbox(entity);
  }
  for (size_t i = 0; i < routers_.size(); i++) {
    const auto& router = routers_.at(i).value();
    result &= router->wait(entity);
  }
  return result;
}

Expected<void> RouterGroup::syncOutbox(const Entity& entity) {
  Expected<void> result = Success;
  for (size_t i = 0; i < routers_.size(); i++) {
    const auto& router = routers_.at(i).value();
    result &= router->syncOutbox(entity);
  }
  return result;
}

Expected<void> RouterGroup::addRouter(Handle<Router> router) {
  auto result = routers_.push_back(std::move(router));
  if (!result) {
    GXF_LOG_WARNING("Failed to add router to group");
    return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE};
  }
  return Success;
}

Expected<void> RouterGroup::removeRouter(Handle<Router> router) {
  for (size_t i = 0; i < routers_.size(); ++i) {
    if (router == routers_.at(i).value()) {
      routers_.erase(i);
      return Success;
    }
  }
  return Unexpected{GXF_ENTITY_COMPONENT_NOT_FOUND};
}

Expected<void> RouterGroup::addNetworkContext(Handle<NetworkContext> context) {
  Expected<void> result = Success;
  for (size_t i = 0; i < routers_.size(); ++i) {
    result &= routers_.at(i).value()->addNetworkContext(context);
  }
  return result;
}

Expected<void> RouterGroup::setClock(Handle<Clock> clock) {
  if (!clock) { return Unexpected{GXF_ARGUMENT_NULL}; }
  clock_ = clock;
  Expected<void> result = Success;
  for (size_t i = 0; i < routers_.size(); i++) {
    const auto& router = routers_.at(i).value();
    result &= router->setClock(clock);
  }
  return result;
}

}  // namespace gxf
}  // namespace nvidia
