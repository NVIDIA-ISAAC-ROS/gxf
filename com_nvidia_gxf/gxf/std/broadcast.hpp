/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_GXF_GXF_STD_BROADCAST_HPP_
#define NVIDIA_GXF_GXF_STD_BROADCAST_HPP_

#include <string.h>
#include <string>

#include "common/fixed_vector.hpp"
#include "gxf/core/component.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

// Mode switch
enum struct BroadcastMode {
  kBroadcast = 0,   // publishes income message to all transmitters;
  kRoundRobin = 1,  // publishes income message to one of transmitters in round robin fashion;
};

// Custom parameter parser for BroadcastMode
template <>
struct ParameterParser<BroadcastMode> {
  static Expected<BroadcastMode> Parse(gxf_context_t context, gxf_uid_t component_uid,
                                       const char* key, const YAML::Node& node,
                                       const std::string& prefix) {
    const std::string value = node.as<std::string>();
    if (strcmp(value.c_str(), "Broadcast") == 0) {
      return BroadcastMode::kBroadcast;
    }
    if (strcmp(value.c_str(), "RoundRobin") == 0) {
      return BroadcastMode::kRoundRobin;
    }
    return Unexpected{GXF_ARGUMENT_OUT_OF_RANGE};
  }
};

// Custom parameter parser for BroadcastMode
template<>
struct ParameterWrapper<BroadcastMode> {
  static Expected<YAML::Node> Wrap(gxf_context_t context, const BroadcastMode& value) {
    YAML::Node node(YAML::NodeType::Scalar);
    switch (value) {
      case BroadcastMode::kBroadcast: {
        node = std::string("Broadcast");
        break;
      }
      case BroadcastMode::kRoundRobin: {
        node = std::string("RoundRobin");
        break;
      }
      default:
        return Unexpected{GXF_PARAMETER_OUT_OF_RANGE};
    }
    return node;
  }
};

// Messages arriving on the input channel are distributed to transmitters.
class Broadcast : public Codelet {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;

  gxf_result_t start() override;

  gxf_result_t tick() override;

 private:
  Parameter<Handle<Receiver>> source_;
  Parameter<BroadcastMode> mode_;

  // Collected transmitters to forward to
  FixedVector<Handle<Transmitter>, kMaxComponents> tx_list_;
  uint64_t round_robin_tx_index_ = 0;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_GXF_STD_BROADCAST_HPP_
