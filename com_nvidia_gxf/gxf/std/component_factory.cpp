/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/component_factory.hpp"

namespace nvidia {
namespace gxf {

Expected<void*> ComponentFactory::allocate(gxf_tid_t tid) {
  void* pointer;
  const gxf_result_t code = allocate_abi(tid, &pointer);
  return ExpectedOrCode(code, pointer);
}

Expected<void> ComponentFactory::deallocate(gxf_tid_t tid, void* pointer) {
  return ExpectedOrCode(deallocate_abi(tid, pointer));
}

}  // namespace gxf
}  // namespace nvidia
