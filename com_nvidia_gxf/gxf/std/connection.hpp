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

 private:
  Parameter<Handle<Transmitter>> source_;
  Parameter<Handle<Receiver>> target_;
};

}  // namespace gxf
}  // namespace nvidia

#endif
