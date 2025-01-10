/*
Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.

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
#include <string>
#include <unordered_map>

#include "gxf/core/entity.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/router.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

// The main class which routes messages from transmitters to receivers. Transmitters and receivers
// can either be connected explicitly with a `Connection` component or implicitly by using a `Topic`
// component.
class MessageRouter : public Router {
 public:
  using CallbackFunction = std::function<void(void*, gxf_uid_t eid)>;

  virtual ~MessageRouter() = default;

  Expected<void> addRoutes(const Entity& entity) override;
  Expected<void> removeRoutes(const Entity& entity) override;
  Expected<void> syncInbox(const Entity& entity) override;
  Expected<void> syncOutbox(const Entity& entity) override;

  // De-/Register a transmitter or receiver for usage with a topic name.
  Expected<void> registerTransmitter(const std::string& topic_name, Handle<Transmitter> tx);
  Expected<void> registerReceiver(const std::string& topic_name, Handle<Receiver> tx);
  Expected<void> deregisterTransmitter(const std::string& topic_name, Handle<Transmitter> tx);
  Expected<void> deregisterReceiver(const std::string& topic_name, Handle<Receiver> tx);

  // @deprecated: Use getConnectedReceivers instead.
  // Gets the receiver which is connected to a transmitter
  Expected<Handle<Receiver>> getRx(Handle<Transmitter> tx);

  // Gets the receivers which are connected to a transmitter
  Expected<std::set<Handle<Receiver>>> getConnectedReceivers(Handle<Transmitter> tx) const;

  // Gets the transmitters which are connected to a receiver.
  Expected<std::set<Handle<Transmitter>>> getConnectedTransmitters(Handle<Receiver> rx) const;

  // Connects a transmitter to a receiver
  Expected<void> connect(Handle<Transmitter> tx, Handle<Receiver> rx);

  // Disconnects the transmitter and receiver
  Expected<void> disconnect(Handle<Transmitter> tx, Handle<Receiver> rx);

  // Sets the clock and network context. Both are dummy functions here.
  Expected<void> setClock(Handle<Clock> clock) override;
  Expected<void> addNetworkContext(Handle<NetworkContext> context) override;

 private:
  // Distributes a message from a transmitter to its connected receivers.
  Expected<void> distribute(Handle<Transmitter> tx, const Entity& message,
                            const std::set<Handle<Receiver>>& receivers);

  // Routes created explicitly with a `Connection` components.
  std::map<Handle<Transmitter>, std::set<Handle<Receiver>>> routes_;
  std::map<Handle<Receiver>, std::set<Handle<Transmitter>>> routes_reversed_;

  // Routes created implicitly with a `Topic` component.
  // Note(lgulich): it would be possible to store both explicit and implicit routes in the
  // containers (by creating on-the-fly-topics for explicit routes) but then explicit connections
  // may have to pay a small performance price for the additional map lookup.
  std::unordered_map<std::string, std::set<Handle<Transmitter>>> topic_and_transmitters_;
  std::unordered_map<std::string, std::set<Handle<Receiver>>> topic_and_receivers_;

  std::map<gxf_uid_t, std::set<Handle<Receiver>>> receivers_;
  std::map<gxf_uid_t, std::set<Handle<Transmitter>>> transmitters_;

  // Maps to do reverse lookup of the topic name of a transmitter/receiver.
  std::map<Handle<Transmitter>, std::string> transmitter_and_topic_;
  std::map<Handle<Receiver>, std::string> receiver_and_topic_;
};

}  // namespace gxf
}  // namespace nvidia

#endif
