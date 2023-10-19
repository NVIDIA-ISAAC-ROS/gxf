/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/connection.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t Connection::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(source_, "source", "Source channel", "");
  result &= registrar->parameter(target_, "target", "Target channel", "");
  return ToResultCode(result);
}

Handle<Transmitter> Connection::source() const {
  return source_;
}
Handle<Receiver> Connection::target() const {
  return target_;
}

}  // namespace gxf
}  // namespace nvidia
