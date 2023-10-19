/*
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef GXF_STD_NETWORK_ROUTER_HPP
#define GXF_STD_NETWORK_ROUTER_HPP


#include "gxf/core/entity.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/std/network_context.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/router.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {

namespace gxf {


class NetworkRouter : public Router {
 public:
  virtual ~NetworkRouter() = default;

  Expected<void> addRoutes(const Entity& entity) override;
  Expected<void> removeRoutes(const Entity& entity) override;
  Expected<void> syncInbox(const Entity& entity) override;
  Expected<void> syncOutbox(const Entity& entity) override;

  // Sets the clock to be used to for updating the pubtime while publishing
  // messages
  Expected<void> setClock(Handle<Clock> clock) override;
  Expected<void> addNetworkContext(Handle<NetworkContext> context) override;

 private:
  Handle<NetworkContext> context_;
};

}  // namespace gxf
}  // namespace nvidia

#endif
