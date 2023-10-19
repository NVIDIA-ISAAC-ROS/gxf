/*
Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef GXF_STD_MESSAGE_ROUTER_HPP
#define GXF_STD_MESSAGE_ROUTER_HPP

#include <map>
#include <set>

#include "gxf/core/entity.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/router.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

// The main class which routes messages from message transmitters to message receivers. It looks
// for entities with components of type Connection, Transmitter and Receiver.
class MessageRouter : public Router {
 public:
  virtual ~MessageRouter() = default;

  Expected<void> addRoutes(const Entity& entity) override;
  Expected<void> removeRoutes(const Entity& entity) override;
  Expected<void> syncInbox(const Entity& entity) override;
  Expected<void> syncOutbox(const Entity& entity) override;

  // Gets the receiver which is connected to a transmitter
  Expected<Handle<Receiver>> getRx(Handle<Transmitter> tx);

  // Connects a transmitter to a receiver
  Expected<void> connect(Handle<Transmitter> tx, Handle<Receiver> rx);

  // Disconnects the transmitter and receiver
  Expected<void> disconnect(Handle<Transmitter> tx, Handle<Receiver> rx);

  // Sets the clock to be used to for updating the pubtime while publishing
  // messages
  Expected<void> setClock(Handle<Clock> clock) override;

  // Sets the network context to be used by network router.
  // Here it is a dummy function
  Expected<void> addNetworkContext(Handle<NetworkContext> context) override;

 private:
  // Distributes a message to a transmitter
  Expected<void> distribute(Handle<Transmitter> tx, const Entity& message);

  // A representation of the message passing graph
  std::map<Handle<Transmitter>, Handle<Receiver>> routes_;
};

}  // namespace gxf
}  // namespace nvidia

#endif
