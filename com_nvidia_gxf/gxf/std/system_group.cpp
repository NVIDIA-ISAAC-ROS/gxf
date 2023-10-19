/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/system_group.hpp"

#include <utility>

namespace nvidia {
namespace gxf {

gxf_result_t SystemGroup::schedule_abi(gxf_uid_t eid) {
  Expected<void> result = Success;
  for (size_t i = 0; i < systems_.size(); i++) {
    const auto& system = systems_.at(i).value();
    result &= ExpectedOrCode(system->schedule_abi(eid));
  }
  return ToResultCode(result);
}

gxf_result_t SystemGroup::unschedule_abi(gxf_uid_t eid) {
  Expected<void> result = Success;
  for (size_t i = 0; i < systems_.size(); i++) {
    const auto& system = systems_.at(i).value();
    result &= ExpectedOrCode(system->unschedule_abi(eid));
  }
  return ToResultCode(result);
}

gxf_result_t SystemGroup::runAsync_abi() {
  Expected<void> result = Success;
  for (size_t i = 0; i < systems_.size(); i++) {
    const auto& system = systems_.at(i).value();
    result &= ExpectedOrCode(system->runAsync_abi());
  }
  return ToResultCode(result);
}

gxf_result_t SystemGroup::stop_abi() {
  Expected<void> result = Success;
  for (size_t i = 0; i < systems_.size(); i++) {
    const auto& system = systems_.at(i).value();
    result &= ExpectedOrCode(system->stop_abi());
  }
  return ToResultCode(result);
}

gxf_result_t SystemGroup::wait_abi() {
  Expected<void> result = Success;
  for (size_t i = 0; i < systems_.size(); i++) {
    const auto& system = systems_.at(i).value();
    result &= ExpectedOrCode(system->wait_abi());
  }
  return ToResultCode(result);
}

Expected<void> SystemGroup::addSystem(Handle<System> system) {
  auto result = systems_.push_back(std::move(system));
  if (!result) {
    GXF_LOG_WARNING("Failed to add system to group");
    return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE};
  }
  return Success;
}

Expected<void> SystemGroup::removeSystem(Handle<System> system) {
  for (size_t i = 0; i < systems_.size(); ++i) {
    if (system == systems_.at(i).value()) {
      systems_.erase(i);
      return Success;
    }
  }
  return Unexpected{GXF_ENTITY_COMPONENT_NOT_FOUND};
}

gxf_result_t SystemGroup::event_notify_abi(gxf_uid_t eid) {
  for (size_t i = 0; i < systems_.size(); i++) {
      const auto& system = systems_.at(i).value();
      const auto& result = system->event_notify_abi(eid);
      if (result != GXF_SUCCESS) { return result; }
  }

  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
