/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/std/system.hpp"

namespace nvidia {
namespace gxf {

Expected<void> System::schedule(const Entity& entity) {
  return ExpectedOrCode(schedule_abi(entity.eid()));
}
Expected<void> System::unschedule(const Entity& entity) {
  return ExpectedOrCode(unschedule_abi(entity.eid()));
}
Expected<void> System::runAsync() {
  return ExpectedOrCode(runAsync_abi());
}
Expected<void> System::stop() {
  return ExpectedOrCode(stop_abi());
}
Expected<void> System::wait() {
  return ExpectedOrCode(wait_abi());
}
Expected<void> System::event_notify(gxf_uid_t eid) {
  return ExpectedOrCode(event_notify_abi(eid));
}

}  // namespace gxf
}  // namespace nvidia
