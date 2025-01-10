/*
Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef GXF_STD_ROUTER_HPP
#define GXF_STD_ROUTER_HPP

#include "gxf/core/entity.hpp"
#include "gxf/std/clock.hpp"
#include "gxf/std/network_context.hpp"

namespace nvidia {
namespace gxf {

// A base class for objects which are routing messages in and out of entities.
class Router {
 public:
  // Notifies the router about a new entity
  virtual Expected<void> addRoutes(const Entity& entity) = 0;

  // Notifies the router about the removal of an entity
  virtual Expected<void> removeRoutes(const Entity& entity) = 0;

  // Synchronizes the inbox of an entity and prepares it for execution
  virtual Expected<void> syncInbox(const Entity& entity) = 0;

  // Synchronizes the inbox of an entity and prepares it for execution
  virtual Expected<void> wait(const Entity& entity) {
    return Success;
  }

  // Synchronizes the outbox of an entity after successful execution
  virtual Expected<void> syncOutbox(const Entity& entity) = 0;

  // Sets the clock to be used to for updating the pubtime while publishing
  // messages
  virtual Expected<void> setClock(Handle<Clock> clock) = 0;

  // Sets the network context to be used by network router
  virtual Expected<void> addNetworkContext(Handle<NetworkContext> context) = 0;

 protected:
  Handle<Clock> clock_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // GXF_STD_ROUTER_HPP
