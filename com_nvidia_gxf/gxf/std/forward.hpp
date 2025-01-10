/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVIDIA_GXF_STD_FORWARD_HPP
#define NVIDIA_GXF_STD_FORWARD_HPP

#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

// Forwards one incoming message at the receiver to the transmitter on each execution
class Forward : public Codelet {
 public:
  gxf_result_t tick() override {
    auto message = in_->receive();
    if (!message) {
      return message.error();
    }
    auto result = out_->publish(message.value());
    return ToResultCode(result);
  }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(in_, "in", "input", "The channel for incoming messages.");
    result &= registrar->parameter(out_, "out", "output", "The channel for outgoing messages");
    return ToResultCode(result);
  }

 private:
  Parameter<Handle<Receiver>> in_;
  Parameter<Handle<Transmitter>> out_;
};

}  // namespace gxf
}  // namespace nvidia

#endif
