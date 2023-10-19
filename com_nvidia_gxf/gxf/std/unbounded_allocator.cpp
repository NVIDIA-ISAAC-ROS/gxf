/*
Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/unbounded_allocator.hpp"

#include <cuda_runtime.h>

#include "common/memory_utils.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t UnboundedAllocator::initialize() {
  return GXF_SUCCESS;
}

gxf_result_t UnboundedAllocator::deinitialize() {
  return GXF_SUCCESS;
}

gxf_result_t UnboundedAllocator::is_available_abi(uint64_t size) {
  // TODO(v2) Is there a way to predict if allocation will fail?
  return GXF_SUCCESS;
}

gxf_result_t UnboundedAllocator::allocate_abi(uint64_t size, int32_t type, void** pointer) {
  if (pointer == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  // We cannot allocate safely a block of size 0.
  // We can artificially increase the size of 1 to remove failure when free_abi is called.
  if (size == 0) {
    size = 1;
  }

  switch (static_cast<MemoryStorageType>(type)) {
    case MemoryStorageType::kHost: {
      const cudaError_t error = cudaMallocHost(pointer, size);
      if (error != cudaSuccess) {
        GXF_LOG_ERROR("Failure in cudaMallocHost. cuda_error: %s, error_str: %s",
                      cudaGetErrorName(error), cudaGetErrorString(error));
        return GXF_OUT_OF_MEMORY;
      }

      // Remember the block so that we know how to delete it
      std::unique_lock<std::mutex> lock(mutex_);
      cuda_host_blocks_.insert(*pointer);
    } break;
    case MemoryStorageType::kDevice: {
      const cudaError_t error = cudaMalloc(pointer, size);
      if (error != cudaSuccess) {
        GXF_LOG_ERROR("Failure in cudaMalloc. cuda_error: %s, error_str: %s",
                      cudaGetErrorName(error), cudaGetErrorString(error));
        return GXF_OUT_OF_MEMORY;
      }

      // Remember the block so that we know how to delete it
      std::unique_lock<std::mutex> lock(mutex_);
      cuda_blocks_.insert(*pointer);
    } break;
    case MemoryStorageType::kSystem: {
      *pointer = AllocateArray<byte>(size);
      if (*pointer == nullptr) {
        return GXF_OUT_OF_MEMORY;
      }
    } break;
    default:
      return GXF_PARAMETER_OUT_OF_RANGE;
  }

  return GXF_SUCCESS;
}

gxf_result_t UnboundedAllocator::free_abi(void* pointer) {
  // Check if this pointer is for a cuda block
  std::unique_lock<std::mutex> lock(mutex_);
  if (cuda_blocks_.count(pointer) == 0) {
    if (cuda_host_blocks_.count(pointer) == 0) {
      DeallocateArray(BytePointer(pointer));
    } else {
      cuda_host_blocks_.erase(pointer);
      const cudaError_t error = cudaFreeHost(pointer);
      if (error != cudaSuccess) {
        GXF_LOG_ERROR("Failure in cudaFreeHost. cuda_error: %s, error_str: %s",
                      cudaGetErrorName(error), cudaGetErrorString(error));
        return GXF_FAILURE;
      }
    }
  } else {
    cuda_blocks_.erase(pointer);
    const cudaError_t error = cudaFree(pointer);
    if (error != cudaSuccess) {
      GXF_LOG_ERROR("Failure in cudaFree. cuda_error: %s, error_str: %s",
                    cudaGetErrorName(error), cudaGetErrorString(error));
      return GXF_FAILURE;
    }
  }

  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
