/*
Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/block_memory_pool.hpp"

#include <cstdint>
#include <memory>

#include "cuda_runtime.h"  // NOLINT

#include "gxf/std/gems/pool/fixed_pool_uint64.hpp"

namespace nvidia {
namespace gxf {

BlockMemoryPool::BlockMemoryPool() {}

BlockMemoryPool::~BlockMemoryPool() {}

gxf_result_t BlockMemoryPool::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      storage_type_, "storage_type", "Storage type",
      "The memory storage type used by this allocator. Can be kHost (0), kDevice (1) or "
      "kSystem (2)", 0);
  result &= registrar->parameter(
      block_size_, "block_size", "Block size",
      "The size of one block of memory in byte. Allocation requests can only be fulfilled if they "
      "fit into one block. If less memory is requested still a full block is issued.");
  result &= registrar->parameter(
      num_blocks_, "num_blocks", "Number of blocks",
      "The total number of blocks which are allocated by the pool. If more blocks are requested "
      "allocation requests will fail.");
  result &= registrar->resource(gpu_device_, "GPU device resource from which allocate CUDA memory");
  return ToResultCode(result);
}

gxf_result_t BlockMemoryPool::initialize() {
  // get device id from GPUDevice Resource
  if (storage_type_.get() == static_cast<int32_t>(MemoryStorageType::kHost) ||
      storage_type_.get() == static_cast<int32_t>(MemoryStorageType::kDevice)) {
    if (gpu_device_.try_get()) {
      dev_id_ = gpu_device_.try_get().value()->device_id();
      GXF_LOG_DEBUG("BlockMemoryPool [cid: %ld]: GPUDevice Resource found. Using dev_id: %d",
        cid(), dev_id_);
    } else {
      dev_id_ = 0;
      GXF_LOG_DEBUG("BlockMemoryPool [cid: %ld]: no GPUDevice Resource found. "
        "Using default device id: %d", cid(), dev_id_);
    }
  }

  std::lock_guard<std::mutex> lock(stack_mutex_);

  const size_t total_size = num_blocks_ * block_size_;

  switch (MemoryStorageType(storage_type_.get())) {
    case MemoryStorageType::kHost: {
      cudaSetDevice(dev_id_);
      const cudaError_t error = cudaMallocHost(&pointer_, total_size);
      if (error != cudaSuccess) {
        GXF_LOG_ERROR("Failure in cudaMallocHost. cuda_error: %s, error_str: %s",
                      cudaGetErrorName(error), cudaGetErrorString(error));
        return GXF_OUT_OF_MEMORY;
      }
    } break;
    case MemoryStorageType::kDevice: {
      cudaSetDevice(dev_id_);
      const cudaError_t error = cudaMalloc(&pointer_, total_size);
      if (error != cudaSuccess) {
        GXF_LOG_ERROR("Failure in cudaMalloc. cuda_error: %s, error_str: %s",
                      cudaGetErrorName(error), cudaGetErrorString(error));
        return GXF_OUT_OF_MEMORY;
      }
    } break;
    case MemoryStorageType::kSystem: {
        pointer_ = new uint8_t[total_size];
        if (pointer_ == nullptr) { return GXF_OUT_OF_MEMORY; }
    } break;
    default:
      return GXF_PARAMETER_OUT_OF_RANGE;
  }

  stack_ = std::make_unique<FixedPoolUint64>();

  const auto result = stack_->allocate(num_blocks_);
  if (!result) {
    GXF_LOG_ERROR("Failed to allocate %lu blocks of memory", num_blocks_.get());
    return GXF_FAILURE;
  }

  stage_ = AllocatorStage::kInitialized;
  return GXF_SUCCESS;
}

gxf_result_t BlockMemoryPool::is_available_abi(uint64_t size) {
  if (stage_ != AllocatorStage::kInitialized) {
    GXF_LOG_ERROR("Allocator must be in Initialized stage before starting."
                  " Current state is %s", allocator_stage_str(stage_));
    return GXF_INVALID_LIFECYCLE_STAGE;
  }
  return size <= (block_size_ * stack_->size()) ? GXF_SUCCESS : GXF_FAILURE;
}

gxf_result_t BlockMemoryPool::allocate_abi(uint64_t size, int32_t type, void** pointer) {
  if (stage_ != AllocatorStage::kInitialized) {
    GXF_LOG_ERROR("Allocator must be in Initialized stage before starting."
                  " Current state is %s", allocator_stage_str(stage_));
    return GXF_INVALID_LIFECYCLE_STAGE;
  }

  if (pointer == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  if (type != storage_type_) {
    return GXF_ARGUMENT_INVALID;
  }
  if (size > block_size_) {
    GXF_LOG_ERROR("Requested %lu bytes of memory in a pool with block size %lu",
                  size, block_size_.get());
    return GXF_ARGUMENT_INVALID;
  }

  std::lock_guard<std::mutex> lock(stack_mutex_);
  if (!stack_) {
    return GXF_CONTRACT_INVALID_SEQUENCE;
  }

  if (!is_available(size)) {
    // too many chunks allocated
    GXF_LOG_ERROR("Too many chunks allocated, memory of size %lu not available", size);
    return GXF_EXCEEDING_PREALLOCATED_SIZE;
  }
  const auto index = stack_->pop();
  if (!index) {
    return GXF_FAILURE;
  }
  *pointer = static_cast<void*>(static_cast<uint8_t*>(pointer_) + index.value() * block_size_);
  return GXF_SUCCESS;
}

gxf_result_t BlockMemoryPool::free_abi(void* void_pointer) {
  uint8_t* pointer = static_cast<uint8_t*>(void_pointer);
  if (pointer < pointer_) {
    // invalid pointer: not part of the pool
    return GXF_ARGUMENT_OUT_OF_RANGE;
  }
  if (!stack_) {
    return GXF_CONTRACT_INVALID_SEQUENCE;
  }

  const uint64_t delta = pointer - static_cast<uint8_t*>(pointer_);
  const uint64_t index = delta / block_size_;
  if (index * block_size_ != delta) {
    // invalid pointer: invalid chunk pointer
    return GXF_ARGUMENT_INVALID;
  }
  {
    std::lock_guard<std::mutex> lock(stack_mutex_);
    if (index >= stack_->capacity()) {
      // invalid pointer: not part of the pool
      return GXF_ARGUMENT_OUT_OF_RANGE;
    }
    const auto result = stack_->push(index);
    if (!result) {
      return GXF_FAILURE;
    }
  }
  return GXF_SUCCESS;
}

gxf_result_t BlockMemoryPool::deinitialize() {
  stack_.release();

  switch (MemoryStorageType(storage_type_.get())) {
    case MemoryStorageType::kHost: {
      const cudaError_t error = cudaFreeHost(pointer_);
      if (error != cudaSuccess) {
        GXF_LOG_ERROR("Failure in cudaFreeHost. cuda_error: %s, error_str: %s",
                      cudaGetErrorName(error), cudaGetErrorString(error));
        return GXF_FAILURE;
      }
    } break;
    case MemoryStorageType::kDevice: {
      const cudaError_t error = cudaFree(pointer_);
      if (error != cudaSuccess) {
        GXF_LOG_ERROR("Failure in cudaFree. cuda_error: %s, error_str: %s",
                      cudaGetErrorName(error), cudaGetErrorString(error));
        return GXF_FAILURE;
      }
    } break;
    case MemoryStorageType::kSystem: {
      delete[] static_cast<uint8_t*>(pointer_);
    } break;
    default:
      return GXF_PARAMETER_OUT_OF_RANGE;
  }

  stage_ = AllocatorStage::kUninitialized;
  return GXF_SUCCESS;
}

uint64_t BlockMemoryPool::block_size_abi() const {
  return block_size_.get();
}

}  // namespace gxf
}  // namespace nvidia
