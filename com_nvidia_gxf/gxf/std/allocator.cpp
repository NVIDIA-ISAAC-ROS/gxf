/*
Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/allocator.hpp"

namespace nvidia {
namespace gxf {

// Returns true if the allocator can provide a memory block with the given size.
bool Allocator::is_available(uint64_t size) {
  return is_available_abi(size) == GXF_SUCCESS;
}

// Allocates a memory block with the given size.
Expected<byte*> Allocator::allocate(uint64_t size, MemoryStorageType type) {
  void* result;
  const gxf_result_t code = allocate_abi(size, static_cast<int32_t>(type), &result);
  return ExpectedOrCode(code, static_cast<byte*>(result));
}

// Frees the given memory block.
Expected<void> Allocator::free(byte* pointer) {
  Expected<void> result =  ExpectedOrCode(free_abi(static_cast<void*>(pointer)));
  GxfEntityNotifyEventType(context(), eid(), GXF_EVENT_MEMORY_FREE);
  return result;
}

// Query block size of this allocator. Defaults to 1 for byte-based allocators.
uint64_t Allocator::block_size_abi() const {
  return 1UL;
}

// Query block size of this allocator.
uint64_t Allocator::block_size() const {
  return block_size_abi();
}

// Get the string value of allocator status
const char*  Allocator::allocator_stage_str(AllocatorStage stage) const {
  switch (stage) {
    GXF_ENUM_TO_STR(AllocatorStage::kUninitialized, Uninitialized)
    GXF_ENUM_TO_STR(AllocatorStage::kInitializationInProgress, InitializationInProgress)
    GXF_ENUM_TO_STR(AllocatorStage::kInitialized, Initialized)
    GXF_ENUM_TO_STR(AllocatorStage::kDeinitializationInProgress, DeinitializationInProgress)
     default:
      return "N/A";
  }
}

}  // namespace gxf
}  // namespace nvidia
