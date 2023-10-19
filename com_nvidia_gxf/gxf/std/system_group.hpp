/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_STD_SYSTEM_GROUP_HPP_
#define NVIDIA_GXF_STD_SYSTEM_GROUP_HPP_

#include "common/fixed_vector.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/std/system.hpp"

namespace nvidia {
namespace gxf {

// A group of systems which implements the System interface and executes called functions for
// all systems which are added to it.
class SystemGroup : public System {
 public:
  gxf_result_t schedule_abi(gxf_uid_t eid) override;
  gxf_result_t unschedule_abi(gxf_uid_t eid) override;
  gxf_result_t runAsync_abi() override;
  gxf_result_t stop_abi() override;
  gxf_result_t wait_abi() override;
  gxf_result_t event_notify_abi(gxf_uid_t eid) override;

  // Adds a system to the group
  Expected<void> addSystem(Handle<System> system);

  // Removes a system from the group
  Expected<void> removeSystem(Handle<System> system);

  // Are there any systems in the group?
  bool empty() const { return systems_.empty(); }

 private:
  FixedVector<Handle<System>, kMaxComponents> systems_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_STD_SYSTEM_GROUP_HPP_
