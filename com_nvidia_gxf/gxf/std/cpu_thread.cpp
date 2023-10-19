/*
Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/cpu_thread.hpp"

#include <utility>

#include "gxf/core/gxf.h"
#include "gxf/core/registrar.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t CPUThread::registerInterface(Registrar* registrar) {
  Expected<void> result;

  result &= registrar->parameter(
    pin_entity_, "pin_entity", "Pin Entity",
    "Set the cpu_core to be pinned to a worker thread or not.",
    false);

  return ToResultCode(result);
}

}  // namespace gxf
}  // namespace nvidia
