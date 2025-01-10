/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GXF_STD_NETWORK_ROUTER_HPP
#define GXF_STD_NETWORK_ROUTER_HPP


#include <set>
#include <unordered_map>

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

  Expected<void> wait(const Entity& entity) override;

  // Sets the clock to be used to for updating the pubtime while publishing
  // messages
  Expected<void> setClock(Handle<Clock> clock) override;
  Expected<void> addNetworkContext(Handle<NetworkContext> context) override;

 private:
  Handle<NetworkContext> context_;
  std::unordered_map<gxf_uid_t, std::set<Handle<Receiver>>> receivers_;
  std::unordered_map<gxf_uid_t, std::set<Handle<Transmitter>>> transmitters_;
};

}  // namespace gxf
}  // namespace nvidia

#endif
