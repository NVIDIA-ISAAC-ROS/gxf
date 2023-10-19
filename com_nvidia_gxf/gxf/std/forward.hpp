/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
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
  gxf_result_t start() override { return GXF_SUCCESS; }

  gxf_result_t tick() override {
    auto message = in_->receive();
    if (!message) {
      return message.error();
    }
    auto result = out_->publish(message.value());
    return ToResultCode(result);
  }

  gxf_result_t stop() override { return GXF_SUCCESS; }

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
