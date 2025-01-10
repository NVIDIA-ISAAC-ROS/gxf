/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_STD_CONNECTION_HPP
#define NVIDIA_GXF_STD_CONNECTION_HPP

#include "gxf/core/component.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

// A component which establishes a connection between two other components.
class Connection : public Component {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;

  Handle<Transmitter> source() const;
  Handle<Receiver> target() const;

  // Set the target parameter
  Expected<void> setReceiver(Handle<Receiver> value) {
    if (!value) {
      GXF_LOG_ERROR("Attempting to set null handle for Rx of Connection component [%s]", name());
      return Unexpected{GXF_ARGUMENT_NULL};
    }
    return target_.set(value);
  }

  // Set the source parameter
  Expected<void> setTransmitter(Handle<Transmitter> value) {
    if (!value) {
      GXF_LOG_ERROR("Attempting to set null handle for Tx of Connection component [%s]", name());
      return Unexpected{GXF_ARGUMENT_NULL};
    }
    return source_.set(value);
  }

 private:
  Parameter<Handle<Transmitter>> source_;
  Parameter<Handle<Receiver>> target_;
};

}  // namespace gxf
}  // namespace nvidia

#endif
