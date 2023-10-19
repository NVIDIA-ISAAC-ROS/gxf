/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/queue.hpp"

namespace nvidia {
namespace gxf {

Expected<Entity> Queue::pop() {
  gxf_uid_t uid;
  const gxf_result_t code = pop_abi(&uid);
  if (code == GXF_SUCCESS) {
    return Entity::Own(context(), uid);
  } else {
    return Unexpected{code};
  }
}

Expected<void> Queue::push(const Entity& other) {
  return ExpectedOrCode(push_abi(other.eid()));
}

Expected<Entity> Queue::peek(int32_t index) {
  gxf_uid_t uid;
  const gxf_result_t code = peek_abi(&uid, index);
  if (code == GXF_SUCCESS) {
    return Entity::Shared(context(), uid);
  } else {
    return Unexpected{code};
  }
}

size_t Queue::capacity() {
  return capacity_abi();
}

size_t Queue::size() {
  return size_abi();
}

}  // namespace gxf
}  // namespace nvidia
