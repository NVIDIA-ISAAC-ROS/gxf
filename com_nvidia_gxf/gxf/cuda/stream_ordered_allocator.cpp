/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/cuda/stream_ordered_allocator.hpp"

#include <stdexcept>
#include <string>
#include <utility>

#include "common/memory_utils.hpp"
#include "gxf/cuda/cuda_common.hpp"
#include "gxf/std/gems/utils/storage_size.hpp"

namespace {
#ifdef __aarch64__
const char* kPoolInitialSize = "8MB";  // 8 MB initial pool size
const char* kPoolMaxSize = "16MB";
#else
const char* kPoolInitialSize = "16MB";  // 16 MB initial pool size
const char* kPoolMaxSize = "32MB";
#endif
const char* release_threshold = "4MB";  // 4MB release threshold

}  // namespace

namespace nvidia {
namespace gxf {

gxf_result_t StreamOrderedAllocator::initialize() {
  try {
    auto device_init_memory_pool_size = GXF_UNWRAP_OR_RETURN(
        StorageSize::ParseStorageSizeString(device_memory_initial_size_.get(), cid()));
    auto device_memory_max_size = GXF_UNWRAP_OR_RETURN(
        StorageSize::ParseStorageSizeString(device_memory_max_size_.get(), cid()));

    auto poolInitialSize =
        GXF_UNWRAP_OR_RETURN(StorageSize::ParseStorageSizeString(kPoolInitialSize, cid()));

      auto poolMaxSize =
        GXF_UNWRAP_OR_RETURN(StorageSize::ParseStorageSizeString(kPoolMaxSize, cid()));

    size_t device_max_memory_pool_size =
        device_init_memory_pool_size != poolInitialSize &&
                device_memory_max_size == poolMaxSize
            ? device_init_memory_pool_size * 2
            : device_memory_max_size;

    pool_props_.allocType = cudaMemAllocationTypePinned;
    pool_props_.location.type = cudaMemLocationTypeDevice;
    pool_props_.maxSize = device_max_memory_pool_size;

    if (gpu_device_.try_get()) {
      pool_props_.location.id = gpu_device_.try_get().value()->device_id();
      GXF_LOG_DEBUG(
          "StreamOrderedAllocator [cid: %ld]: GPUDevice Resource found. Using dev_id: "
          "%d",
          cid(), pool_props_.location.id);
    } else {
      pool_props_.location.id = 0;
    }

    CHECK_CUDA_ERROR_RESULT(cudaMemPoolCreate(&memory_pool_, &pool_props_),
                            "Failed to create cuda memory pool");

    auto threshold =
        GXF_UNWRAP_OR_RETURN(StorageSize::ParseStorageSizeString(release_threshold_.get(), cid()));

    CHECK_CUDA_ERROR_RESULT(
        cudaMemPoolSetAttribute(memory_pool_, cudaMemPoolAttrReleaseThreshold, &(threshold)),
        "Failed to set cuda release threshold to pool");

    CHECK_CUDA_ERROR_RESULT(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                            "Failed to create cuda stream");
    stage_ = AllocatorStage::kInitialized;

    // Allocate and immediately deallocate the initial_pool_size to prime the pool with the
    // specified size
    uint8_t* mem_pointer;
    if (allocate_abi(device_init_memory_pool_size, 1, reinterpret_cast<void**>(&mem_pointer)) !=
        GXF_SUCCESS) {
      GXF_LOG_ERROR("Unexpected error while initializing stream ordered Allocator %s: %s", name(),
                    "The initial memory pool allocation was unsuccessful.");
      return GXF_FAILURE;
    }

    return free_abi(mem_pointer);
  } catch (const std::exception& e) {
    GXF_LOG_ERROR("Unexpected error while initializing Stream Ordered Allocator %s: %s", name(),
                  e.what());
  }
  return GXF_FAILURE;
}

gxf_result_t StreamOrderedAllocator::deinitialize() {
  stage_ = AllocatorStage::kUninitialized;
  if (!pool_map_.empty()) {
    GXF_LOG_WARNING("StreamOrderedAllocator pool %s still has unreleased memory", name());
  }
  CHECK_CUDA_ERROR_RESULT(cudaStreamSynchronize(stream_), "Failed to synchronize cuda stream");
  CHECK_CUDA_ERROR_RESULT(cudaMemPoolDestroy(memory_pool_), "Failed to destroy cuda memory pool");
  CHECK_CUDA_ERROR_RESULT(cudaStreamDestroy(stream_), "Failed to destroy cuda stream");
  stream_ = nullptr;
  return GXF_SUCCESS;
}

gxf_result_t StreamOrderedAllocator::registerInterface(Registrar* registrar) {
  Expected<void> result;
  registrar->registerParameterlessComponent();
  result &= registrar->resource(gpu_device_, "GPU device resource from which allocate CUDA memory");
  result &= registrar->parameter(
      release_threshold_, "release_threshold",
      "Amount of reserved memory to hold onto before trying to release memory back to the "
      "OS",
      "The release threshold specifies the maximum amount of memory the pool caches. Units can be "
      "B, KB, MB, GB, TB",
      std::string(release_threshold));
  result &= registrar->parameter(
      device_memory_initial_size_, "device_memory_initial_size",
      "Device Memory Pool Initial Size.",
      "The initial memory pool size used by this device. Units can be B, KB, MB, GB, TB",
      std::string(kPoolInitialSize));
  result &= registrar->parameter(
      device_memory_max_size_, "device_memory_max_size", "Device Memory Pool Maximum Size",
      "The max memory pool size used by this device. Units can be B, KB, MB, GB, TB",
      std::string(kPoolMaxSize));
  return ToResultCode(result);
}

gxf_result_t StreamOrderedAllocator::is_available_abi(uint64_t size) {
  // TODO(v2) Is there a way to predict if allocation will fail?
  if (stage_ != AllocatorStage::kInitialized) {
    GXF_LOG_ERROR(
        "Allocator must be in Initialized stage before starting."
        " Current state is %s",
        allocator_stage_str(stage_));
    return GXF_INVALID_LIFECYCLE_STAGE;
  }

  size_t usedMemory, reservedMemory;
  CHECK_CUDA_ERROR_RESULT(
      cudaMemPoolGetAttribute(memory_pool_, cudaMemPoolAttrUsedMemCurrent, &usedMemory),
      "Failed to get total used memory size from the pool.");
  CHECK_CUDA_ERROR_RESULT(
      cudaMemPoolGetAttribute(memory_pool_, cudaMemPoolAttrReservedMemHigh, &reservedMemory),
      "Failed to get reserved memory size from the pool.");
  const size_t availableBytes = reservedMemory - usedMemory;
  return size <= availableBytes ? GXF_SUCCESS : GXF_FAILURE;
}

gxf_result_t StreamOrderedAllocator::allocate_abi(uint64_t size, int32_t type, void** pointer) {
  if (stage_ != AllocatorStage::kInitialized) {
    GXF_LOG_ERROR(
        "Allocator must be in Initialized stage before starting."
        " Current state is %s",
        allocator_stage_str(stage_));
    return GXF_INVALID_LIFECYCLE_STAGE;
  }

  if (pointer == nullptr) {
    return GXF_ARGUMENT_NULL;
  }

  // We cannot allocate safely a block of size 0.
  // We can artificially increase the size of 1 to remove failure when free_abi
  // is called.
  if (size == 0) {
    size = 1;
  }

  if (type != static_cast<int32_t>(MemoryStorageType::kDevice)) {
    GXF_LOG_ERROR("Only Device memory type is supported in StreamOrderedAllocator [%05ld]('%s').",
                  eid(), name());
    return GXF_ARGUMENT_INVALID;
  }

  CHECK_CUDA_ERROR_RESULT(cudaMallocFromPoolAsync(pointer, size, memory_pool_, stream_),
                          "Failed to allocate memory from a cuda allocator");
  CHECK_CUDA_ERROR_RESULT(cudaStreamSynchronize(stream_),
                          "Failed to synchronize a cuda stream");

  pool_map_.emplace(*pointer, size);
  return GXF_SUCCESS;
}

gxf_result_t StreamOrderedAllocator::allocate_async_abi(uint64_t size, void** pointer,
                                               cudaStream_t stream) {
  if (stage_ != AllocatorStage::kInitialized) {
    GXF_LOG_ERROR(
        "Allocator must be in Initialized stage before starting."
        " Current state is %s",
        allocator_stage_str(stage_));
    return GXF_INVALID_LIFECYCLE_STAGE;
  }

  if (pointer == nullptr) {
    return GXF_ARGUMENT_NULL;
  }

  // We cannot allocate safely a block of size 0.
  // We can artificially increase the size of 1 to remove failure when free_abi
  // is called.
  if (size == 0) {
    size = 1;
  }

  CHECK_CUDA_ERROR_RESULT(cudaMallocFromPoolAsync(pointer, size, memory_pool_, stream),
                          "Failed to allocate memory from a cuda allocator");

  pool_map_.emplace(*pointer, size);
  return GXF_SUCCESS;
}

gxf_result_t StreamOrderedAllocator::free_async_abi(void* pointer, cudaStream_t stream) {
  const auto it = pool_map_.find(pointer);
  if (it != pool_map_.end()) {
    CHECK_CUDA_ERROR_RESULT(cudaFreeAsync(pointer, stream), "Failed to free cuda memory");
    pool_map_.erase(pointer);
    return GXF_SUCCESS;
  }

  GXF_LOG_ERROR("The provided memory pointer is not defined within this memory pool [%05ld]('%s').",
                eid(), name());
  return GXF_FAILURE;
}

gxf_result_t StreamOrderedAllocator::free_abi(void* pointer) {
  const auto it = pool_map_.find(pointer);
  if (it != pool_map_.end()) {
    if (stream_) {
      CHECK_CUDA_ERROR_RESULT(cudaFreeAsync(pointer, stream_), "Failed to free cuda memory");
      CHECK_CUDA_ERROR_RESULT(cudaStreamSynchronize(stream_), "Failed to synchronize cuda stream");
    }
    pool_map_.erase(pointer);
    return GXF_SUCCESS;
  }
  GXF_LOG_ERROR("The provided memory pointer is not defined within this memory pool [%05ld]('%s').",
                eid(), name());
  return GXF_FAILURE;
}

Expected<size_t> StreamOrderedAllocator::get_pool_size(MemoryStorageType type) const {
  if (type != MemoryStorageType::kDevice) {
    GXF_LOG_ERROR("Only Device memory type is supported in StreamOrderedAllocator [%05ld]('%s').",
                  eid(), name());
    return GXF_ARGUMENT_INVALID;
  }

  size_t currentSize;
  CHECK_CUDA_ERROR_RESULT(
      cudaMemPoolGetAttribute(memory_pool_, cudaMemPoolAttrUsedMemCurrent, &currentSize),
      "Failed to get current pool size");
  return currentSize;
}

}  // namespace gxf
}  // namespace nvidia
