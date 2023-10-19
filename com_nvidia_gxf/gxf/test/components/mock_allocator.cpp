/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "gxf/test/components/mock_allocator.hpp"

#include <string>
#include <unordered_map>

#include "common/memory_utils.hpp"
#include "cuda_runtime.h"

namespace nvidia {
namespace gxf {
namespace test {

namespace {

constexpr size_t kPetabyte = 1UL << 50;
constexpr size_t kTerabyte = 1UL << 40;
constexpr size_t kGigabyte = 1UL << 30;
constexpr size_t kMegabyte = 1UL << 20;
constexpr size_t kKilobyte = 1UL << 10;

std::string BytesToString(size_t bytes) {
  std::string string;
  if (bytes > kPetabyte) {
    string = std::to_string(bytes / kPetabyte) + "PB";
  } else if (bytes > kTerabyte) {
    string = std::to_string(bytes / kTerabyte) + "TB";
  } else if (bytes > kGigabyte) {
    string = std::to_string(bytes / kGigabyte) + "GB";
  } else if (bytes > kMegabyte) {
    string = std::to_string(bytes / kMegabyte) + "MB";
  } else if (bytes > kKilobyte) {
    string = std::to_string(bytes / kKilobyte) + "kB";
  } else {
    string = std::to_string(bytes) + "B";
  }
  return string;
}

Expected<void*> Allocate(size_t size, MemoryStorageType storage_type) {
  void* pointer;
  switch (storage_type) {
    case MemoryStorageType::kHost: {
      const cudaError_t error = cudaMallocHost(&pointer, size);
      if (error != cudaSuccess) {
        GXF_LOG_ERROR("Failure in cudaMallocHost. cuda_error: %s, error_str: %s",
                      cudaGetErrorName(error), cudaGetErrorString(error));
        return Unexpected{GXF_OUT_OF_MEMORY};
      }
    } break;
    case MemoryStorageType::kDevice: {
      const cudaError_t error = cudaMalloc(&pointer, size);
      if (error != cudaSuccess) {
        GXF_LOG_ERROR("Failure in cudaMalloc. cuda_error: %s, error_str: %s",
                      cudaGetErrorName(error), cudaGetErrorString(error));
        return Unexpected{GXF_OUT_OF_MEMORY};
      }
    } break;
    case MemoryStorageType::kSystem: {
      pointer = AllocateArray<byte>(size);
      if (pointer == nullptr) {
        return Unexpected{GXF_OUT_OF_MEMORY};
      }
    } break;
    default:
      return Unexpected{GXF_ARGUMENT_INVALID};
  }
  return pointer;
}

Expected<void> Deallocate(void* pointer, MemoryStorageType storage_type) {
  switch (storage_type) {
    case MemoryStorageType::kHost: {
      const cudaError_t error = cudaFreeHost(pointer);
      if (error != cudaSuccess) {
        GXF_LOG_ERROR("Failure in cudaFreeHost. cuda_error: %s, error_str: %s",
                      cudaGetErrorName(error), cudaGetErrorString(error));
        return Unexpected{GXF_FAILURE};
      }
    } break;
    case MemoryStorageType::kDevice: {
      const cudaError_t error = cudaFree(pointer);
      if (error != cudaSuccess) {
        GXF_LOG_ERROR("Failure in cudaFree. cuda_error: %s, error_str: %s",
                      cudaGetErrorName(error), cudaGetErrorString(error));
        return Unexpected{GXF_FAILURE};
      }
    } break;
    case MemoryStorageType::kSystem: {
      DeallocateArray(BytePointer(pointer));
    } break;
    default:
      return Unexpected{GXF_ARGUMENT_INVALID};
  }
  return Success;
}

}  // namespace

gxf_result_t MockAllocator::registerInterface(Registrar* registrar) {
  if (registrar == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  Expected<void> result;
  result &= registrar->parameter(
      ignore_memory_leak_, "ignore_memory_leak", "Ignore Memory Leak",
      "Does not raise an error if there is a memory leak",
      false);
  result &= registrar->parameter(
      fail_on_free_, "fail_on_free", "Fail On Free",
      "Forces free_abi() to return GXF_FAILURE",
      false);
  result &= registrar->parameter(
      max_block_size_, "max_block_size", "Max Block Size",
      "Maximum memory block size that can be allocated",
      Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      max_host_allocation_, "max_host_allocation", "Max Host Allocation",
      "Maximum amount of host memory that can be allocated",
      Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      max_device_allocation_, "max_device_allocation", "Max Device Allocation",
      "Maximum amount of device memory that can be allocated",
      Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      max_system_allocation_, "max_system_allocation", "Max System Allocation",
      "Maximum amount of system memory that can be allocated",
      Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  return ToResultCode(result);
}

gxf_result_t MockAllocator::initialize() {
  memory_blocks_.clear();
  metrics_ = {
    { MemoryStorageType::kHost,   Metrics{0, 0} },
    { MemoryStorageType::kDevice, Metrics{0, 0} },
    { MemoryStorageType::kSystem, Metrics{0, 0} },
  };
  stage_ = AllocatorStage::kInitialized;
  return GXF_SUCCESS;
}

gxf_result_t MockAllocator::deinitialize() {
  printMemoryUsage();
  auto result = checkForMemoryLeaks();
  if (!ignore_memory_leak_ && !result) {
    return ToResultCode(result);
  }
  stage_ = AllocatorStage::kUninitialized;
  return GXF_SUCCESS;
}

gxf_result_t MockAllocator::is_available_abi(uint64_t size) {
  if (stage_ != AllocatorStage::kInitialized) {
    GXF_LOG_ERROR("Allocator must be in Initialized stage before starting."
                  " Current state is %s", allocator_stage_str(stage_));
    return GXF_INVALID_LIFECYCLE_STAGE;
  }
  const auto& max_block_size = max_block_size_.try_get();
  if (max_block_size && size > max_block_size.value()) {
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t MockAllocator::allocate_abi(uint64_t size, int32_t type, void** pointer) {
  if (stage_ != AllocatorStage::kInitialized) {
    GXF_LOG_ERROR("Allocator must be in Initialized stage before starting."
                  " Current state is %s", allocator_stage_str(stage_));
    return GXF_INVALID_LIFECYCLE_STAGE;
  }
  if (pointer == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  const MemoryStorageType storage_type = static_cast<MemoryStorageType>(type);
  return ToResultCode(
      verifyAllocation(size, storage_type)
      .and_then([&]() { return Allocate(size, storage_type); })
      .map([&](void* address) {
        std::unique_lock<std::shared_timed_mutex> lock(mutex_);
        memory_blocks_[address] = MemoryBlock{storage_type, size};
        metrics_[storage_type].blocks++;
        metrics_[storage_type].allocation += size;
        *pointer = address;
        return Success;
      }));
}

gxf_result_t MockAllocator::free_abi(void* pointer) {
  if (fail_on_free_) {
    return GXF_FAILURE;
  }
  return ToResultCode(
      verifyDeallocation(pointer)
      .map([&](MemoryStorageType storage_type) {
        std::unique_lock<std::shared_timed_mutex> lock(mutex_);
        memory_blocks_.erase(pointer);
        return Deallocate(pointer, storage_type);
      }));
}

Expected<void> MockAllocator::verifyAllocation(size_t size, MemoryStorageType storage_type) {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  if (!is_available(size)) {
    return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE};
  }
  switch (storage_type) {
    case MemoryStorageType::kHost: {
      const auto& max_host_allocation = max_host_allocation_.try_get();
      const size_t host_allocation = metrics_[MemoryStorageType::kHost].allocation + size;
      if (max_host_allocation && host_allocation > max_host_allocation.value()) {
        return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE};
      }
    } break;
    case MemoryStorageType::kDevice: {
      const auto& max_device_allocation = max_device_allocation_.try_get();
      const size_t device_allocation = metrics_[MemoryStorageType::kDevice].allocation + size;
      if (max_device_allocation && device_allocation > max_device_allocation.value()) {
        return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE};
      }
    } break;
    case MemoryStorageType::kSystem: {
      const auto& max_system_allocation = max_system_allocation_.try_get();
      const size_t system_allocation = metrics_[MemoryStorageType::kSystem].allocation + size;
      if (max_system_allocation && system_allocation > max_system_allocation.value()) {
        return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE};
      }
    } break;
    default:
      return Unexpected{GXF_ARGUMENT_INVALID};
  }
  return Success;
}

Expected<MemoryStorageType> MockAllocator::verifyDeallocation(void* pointer) {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  const auto iter = memory_blocks_.find(pointer);
  if (iter == memory_blocks_.end()) {
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  const auto& block = iter->second;
  return block.storage_type;
}

Expected<void> MockAllocator::checkForMemoryLeaks() {
  std::unordered_map<MemoryStorageType, Metrics> lost = {
    { MemoryStorageType::kHost,   Metrics{0, 0} },
    { MemoryStorageType::kDevice, Metrics{0, 0} },
    { MemoryStorageType::kSystem, Metrics{0, 0} },
  };
  for (const auto& entry : memory_blocks_) {
    const auto& block = entry.second;
    lost[block.storage_type].blocks++;
    lost[block.storage_type].allocation += block.size;
  }
  size_t lost_blocks = 0;
  size_t lost_allocation = 0;
  for (const auto& entry : lost) {
    const auto& metrics = entry.second;
    lost_blocks += metrics.blocks;
    lost_allocation += metrics.allocation;
  }
  const auto& host = lost[MemoryStorageType::kHost];
  const auto& device = lost[MemoryStorageType::kDevice];
  const auto& system = lost[MemoryStorageType::kSystem];
  if (lost_blocks > 0) {
    GXF_LOG_WARNING("[%s/%s] Blocks Lost: %zu (%s) "
                    "{ Host: %zu (%s) | Device: %zu (%s) | System: %zu (%s) }",
                    entity().name(), name(),
                    lost_blocks, BytesToString(lost_allocation).c_str(),
                    host.blocks, BytesToString(host.allocation).c_str(),
                    device.blocks, BytesToString(device.allocation).c_str(),
                    system.blocks, BytesToString(system.allocation).c_str());
    return Unexpected{GXF_FAILURE};
  }
  return Success;
}

void MockAllocator::printMemoryUsage() {
  size_t total_blocks = 0;
  size_t total_allocation = 0;
  for (const auto& entry : metrics_) {
    const auto& metrics = entry.second;
    total_blocks += metrics.blocks;
    total_allocation += metrics.allocation;
  }
  const auto& host = metrics_[MemoryStorageType::kHost];
  const auto& device = metrics_[MemoryStorageType::kDevice];
  const auto& system = metrics_[MemoryStorageType::kSystem];
  GXF_LOG_INFO("[%s/%s] Blocks Allocated: %zu (%s) "
               "{ Host: %zu (%s) | Device: %zu (%s) | System: %zu (%s) }",
               entity().name(), name(),
               total_blocks, BytesToString(total_allocation).c_str(),
               host.blocks, BytesToString(host.allocation).c_str(),
               device.blocks, BytesToString(device.allocation).c_str(),
               system.blocks, BytesToString(system.allocation).c_str());
}

}  // namespace test
}  // namespace gxf
}  // namespace nvidia
