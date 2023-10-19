/*
Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_STD_ROUTER_GROUP_HPP_
#define NVIDIA_GXF_STD_ROUTER_GROUP_HPP_

#include "common/fixed_vector.hpp"
#include "gxf/core/entity.hpp"
#include "gxf/std/network_context.hpp"
#include "gxf/std/router.hpp"

namespace nvidia {
namespace gxf {

// A group of routers which are executed sequentially
class RouterGroup : public Router {
 public:
  virtual ~RouterGroup() = default;

  Expected<void> addRoutes(const Entity& entity) override;
  Expected<void> removeRoutes(const Entity& entity) override;
  Expected<void> syncInbox(const Entity& entity) override;
  Expected<void> syncOutbox(const Entity& entity) override;

  // Sets the clock in all the routers in the routers_
  // to be used to for updating the pubtime while publishing
  // messages
  Expected<void> setClock(Handle<Clock> clock) override;

  // Adds a router to the group
  Expected<void> addRouter(Handle<Router> router);

  // Removes a router from the group
  Expected<void> removeRouter(Handle<Router> router);

  // Adds network context to the network router
  Expected<void> addNetworkContext(Handle<NetworkContext> context) override;

 private:
  FixedVector<Handle<Router>, kMaxComponents> routers_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_STD_ROUTER_GROUP_HPP_
