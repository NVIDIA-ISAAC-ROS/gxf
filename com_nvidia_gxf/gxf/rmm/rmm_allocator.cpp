/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/rmm/rmm_allocator.hpp"

#include <memory>
#include <string>
#include <utility>

#include "common/memory_utils.hpp"
#include "gxf/cuda/cuda_common.hpp"
#include "gxf/std/gems/utils/storage_size.hpp"
#include "rmm/mr/device/per_device_resource.hpp"

namespace {
constexpr size_t calculate_alignUp_buffer(size_t current_pool_size) {
  // Use alignUp to ensure the size is a multiple of 256 bytes
  return rmm::align_up(current_pool_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

#ifdef __aarch64__
const char* kPoolInitialSize = "8MB";  // 8 MB initial pool size
const char* kPoolMaxSize = "16MB";
#else
const char* kPoolInitialSize = "16MB";  // 16 MB initial pool size
const char* kPoolMaxSize = "32MB";
#endif

}  // namespace

namespace nvidia {
namespace gxf {

gxf_result_t RMMAllocator::initialize() {
  try {
    CHECK_CUDA_ERROR_RESULT(cudaStreamCreate(&stream_), "Failed to create cuda stream");

    size_t device_memory_initial_size = GXF_UNWRAP_OR_RETURN(
        StorageSize::ParseStorageSizeString(device_memory_initial_size_.get(), cid()));
    size_t host_memory_initial_size = GXF_UNWRAP_OR_RETURN(
        StorageSize::ParseStorageSizeString(host_memory_initial_size_.get(), cid()));

    size_t device_init_memory_pool_size = calculate_alignUp_buffer(device_memory_initial_size);
    size_t host_init_memory_pool_size = calculate_alignUp_buffer(host_memory_initial_size);

    size_t poolInitialSize =
        GXF_UNWRAP_OR_RETURN(StorageSize::ParseStorageSizeString(kPoolInitialSize, cid()));
    size_t poolMaxSize =
        GXF_UNWRAP_OR_RETURN(StorageSize::ParseStorageSizeString(kPoolMaxSize, cid()));
    size_t device_memory_max_size = GXF_UNWRAP_OR_RETURN(
        StorageSize::ParseStorageSizeString(device_memory_max_size_.get(), cid()));
    size_t host_memory_max_size = GXF_UNWRAP_OR_RETURN(
        StorageSize::ParseStorageSizeString(host_memory_max_size_.get(), cid()));

    device_max_memory_pool_size_ =
        device_init_memory_pool_size != poolInitialSize && device_memory_max_size == poolMaxSize
            ? calculate_alignUp_buffer(device_init_memory_pool_size * 2)
            : calculate_alignUp_buffer(device_memory_max_size);

    host_max_memory_pool_size_ =
        host_init_memory_pool_size != poolInitialSize && host_memory_max_size == poolMaxSize
            ? calculate_alignUp_buffer(host_init_memory_pool_size * 2)
            : calculate_alignUp_buffer(host_memory_max_size);

    // Create a pool memory resource for device memory
    device_mr = std::make_unique<rmm::mr::cuda_memory_resource>();
    pool_mr_device = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
        device_mr.get(), device_init_memory_pool_size, device_max_memory_pool_size_);

    // Create a pinned memory resource for host memory
    pinned_mr = std::make_unique<rmm::mr::pinned_memory_resource>();
    pool_mr_host = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::pinned_memory_resource>>(
        pinned_mr.get(), host_init_memory_pool_size, host_max_memory_pool_size_);

    stage_ = AllocatorStage::kInitialized;
    return GXF_SUCCESS;
  } catch (const std::exception& e) {
    GXF_LOG_ERROR("Unexpected error while initializing RMM Allocator %s: %s", name(), e.what());
  }
  return GXF_FAILURE;
}

gxf_result_t RMMAllocator::deinitialize() {
  stage_ = AllocatorStage::kUninitialized;
  if (!pool_map.empty()) {
    GXF_LOG_WARNING("RMMAllocator pool %s still has unreleased memory", name());
  }
  CHECK_CUDA_ERROR_RESULT(cudaStreamDestroy(stream_), "Failed to destroy stream");
  stream_ = nullptr;
  return GXF_SUCCESS;
}

gxf_result_t RMMAllocator::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      device_memory_initial_size_, "device_memory_initial_size", "Device Memory Pool Initial Size",
      "The initial memory pool size used by this device in MB. Units can be B, KB, MB, GB, TB.",
      std::string(kPoolInitialSize));
  result &= registrar->parameter(
      device_memory_max_size_, "device_memory_max_size", "Device Memory Pool Maximum Size",
      "The max memory pool size used by this device. Units can be B, KB, MB, GB, TB.",
      std::string(kPoolMaxSize));
  result &= registrar->parameter(
      host_memory_initial_size_, "host_memory_initial_size", "Host Memory Pool Initial Size",
      "The initial memory pool size used by this host. Units can be B, KB, MB, GB, TB.",
      std::string(kPoolInitialSize));
  result &= registrar->parameter(
      host_memory_max_size_, "host_memory_max_size", "Host Memory Pool Maximum Size",
      "The max memory pool size used by this host. Units can be B, KB, MB, GB, TB.",
      std::string(kPoolMaxSize));
  result &= registrar->resource(gpu_device_, "GPU device resource from which allocate CUDA memory");
  return ToResultCode(result);
}

gxf_result_t RMMAllocator::is_available_abi(uint64_t size) {
  if (stage_ != AllocatorStage::kInitialized) {
    GXF_LOG_ERROR(
        "Allocator must be in Initialized stage before starting."
        " Current state is %s",
        allocator_stage_str(stage_));
    return GXF_INVALID_LIFECYCLE_STAGE;
  }
  GXF_LOG_WARNING(
      "The RMM allocator does not support this API. It's recommended to use is_rmm_available_abi "
      "instead.");
  return GXF_SUCCESS;
}

gxf_result_t RMMAllocator::is_rmm_available_abi(uint64_t size, MemoryStorageType type) {
  if (type == MemoryStorageType::kSystem) {
    GXF_LOG_ERROR("The storage type provided is not supported in RMM Component [%05ld]('%s')",
                  eid(), name());
    return GXF_ARGUMENT_INVALID;
  }

  size_t availableBytes = 0;
  if (type == MemoryStorageType::kDevice) {
    availableBytes = device_max_memory_pool_size_ - pool_mr_device->pool_size();
  } else if (type == MemoryStorageType::kHost) {
    availableBytes = host_max_memory_pool_size_ - pool_mr_host->pool_size();
  }

  return size <= availableBytes ? GXF_SUCCESS : GXF_FAILURE;
}

gxf_result_t RMMAllocator::allocate_abi(uint64_t size, int32_t type, void** pointer) {
  try {
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

    if (type == static_cast<int32_t>(MemoryStorageType::kSystem)) {
      GXF_LOG_ERROR("The storage type provided is not supported in RMM Component [%05ld]('%s')",
                    eid(), name());
      return GXF_ARGUMENT_INVALID;
    }

    auto mem_type = static_cast<MemoryStorageType>(type);
    if (mem_type == MemoryStorageType::kDevice) {
      *pointer = pool_mr_device->allocate(size, stream_);
    } else {
      *pointer = pool_mr_host->allocate(size);
    }

    pool_map.emplace(*pointer, std::make_pair(std::size_t(size), mem_type));

    return GXF_SUCCESS;
  } catch (const std::exception& e) {
    GXF_LOG_ERROR("Unexpected error while allocating memory [%05ld]('%s') : %s", eid(), name(),
                  e.what());
  }

  return GXF_FAILURE;
}

gxf_result_t RMMAllocator::allocate_async_abi(uint64_t size, void** pointer,
                                              cudaStream_t stream) {
  try {
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

    *pointer = pool_mr_device->allocate_async(size, stream);

    pool_map.emplace(*pointer,
                     std::make_pair(std::size_t(size), MemoryStorageType::kDevice));

    return GXF_SUCCESS;
  } catch (const std::exception& e) {
    GXF_LOG_ERROR("Unexpected error while allocating async memory [%05ld]('%s') : %s", eid(),
                  name(), e.what());
  }

  return GXF_FAILURE;
}

gxf_result_t RMMAllocator::free_async_abi(void* pointer, cudaStream_t stream) {
  try {
    const auto it = pool_map.find(pointer);
    if (it != pool_map.end()) {
      auto mem_type = it->second.second;
      if (mem_type == MemoryStorageType::kDevice) {
        auto size = it->second.first;
        pool_mr_device->deallocate_async(pointer, size, stream);
        pool_map.erase(pointer);
        return GXF_SUCCESS;
      } else {
        GXF_LOG_ERROR("The provided memory pointer should be allocated in device memory.");
      }
    }
    GXF_LOG_ERROR(
        "The provided memory pointer is not defined within this memory pool [%05ld]('%s')",
        eid(), name());
  } catch (const std::exception& e) {
    GXF_LOG_ERROR("Unexpected error during deallocation [%05ld]('%s'): %s", eid(), name(),
                  e.what());
  }

  return GXF_FAILURE;
}

gxf_result_t RMMAllocator::free_abi(void* pointer) {
  try {
    const auto it = pool_map.find(pointer);
    if (it != pool_map.end()) {
      auto size = it->second.first;
      auto mem_type = it->second.second;
      if (mem_type == MemoryStorageType::kDevice && stream_) {
        pool_mr_device->deallocate(pointer, size, stream_);
      } else {
        pool_mr_host->deallocate(pointer, size);
      }
      pool_map.erase(pointer);
      return GXF_SUCCESS;
    }
    GXF_LOG_ERROR(
        "The provided memory pointer is not defined within this memory pool [%05ld]('%s')",
        eid(), name());
  } catch (const std::exception& e) {
    GXF_LOG_ERROR("Unexpected error during deallocation [%05ld]('%s'): %s", eid(), name(),
                  e.what());
  }

  return GXF_FAILURE;
}

Expected<size_t> RMMAllocator::get_pool_size(MemoryStorageType type) const {
  if (type == MemoryStorageType::kSystem) {
    GXF_LOG_ERROR("The storage type provided is not supported in RMM Component [%05ld]('%s')",
                  eid(), name());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  size_t currentSize = 0;
  if (type == MemoryStorageType::kDevice) {
    currentSize = pool_mr_device->pool_size();
  } else if (type == MemoryStorageType::kHost) {
    currentSize = pool_mr_host->pool_size();
  }
  return currentSize;
}

}  // namespace gxf
}  // namespace nvidia
