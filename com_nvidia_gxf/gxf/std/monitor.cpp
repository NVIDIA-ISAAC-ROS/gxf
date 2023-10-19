/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/monitor.hpp"

namespace nvidia {
namespace gxf {

Expected<void> Monitor::onExecute(gxf_uid_t eid, uint64_t timestamp, gxf_result_t code) {
  return ExpectedOrCode(on_execute_abi(eid, timestamp, code));
}

}  // namespace gxf
}  // namespace nvidia
