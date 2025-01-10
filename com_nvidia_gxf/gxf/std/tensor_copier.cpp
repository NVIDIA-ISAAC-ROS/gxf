/*
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/tensor_copier.hpp"

#include <cuda_runtime.h>

#include <utility>

#include "gxf/std/tensor.hpp"

namespace nvidia {
namespace gxf {

namespace {

enum struct CopyMode {
  kCopyToDevice = 0,  // Copy to device memory only
  kCopyToHost = 1,    // Copy to host memory only
  kCopyToSystem = 2,  // Copy to system memory only
};

}  // namespace

gxf_result_t TensorCopier::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
    receiver_, "receiver", "Receiver",
    "Receiver for incoming entities");
  result &= registrar->parameter(
    transmitter_, "transmitter", "Transmitter",
    "Transmitter for outgoing entities ");
  result &= registrar->parameter(
    allocator_, "allocator", "Allocator",
    "Memory allocator for tensor data");
  result &= registrar->parameter(
    mode_, "mode", "Copy mode",
    "Configuration to select what tensors to copy - "
    "kCopyToDevice (0) - copies to device memory, ignores device allocation; "
    "kCopyToHost (1) - copies to pinned host memory, ignores host allocation; "
    "kCopyToSystem (2) - copies to system memory, ignores system allocation");
  return ToResultCode(result);
}

gxf_result_t TensorCopier::tick() {
  Expected<Entity> entity = receiver_->receive();
  if (!entity) {
    return ToResultCode(entity);
  }
  auto tensors = entity->findAllHeap<Tensor>();
  if (!tensors) {
    return ToResultCode(tensors);
  }
  for (auto tensor : tensors.value()) {
    if (!tensor) {
      GXF_LOG_ERROR("Found a bad tensor while copying");
      return GXF_FAILURE;
    }
    cudaMemcpyKind operation;
    MemoryStorageType storage_type;

    if (CopyMode(mode_.get()) == CopyMode::kCopyToDevice) {
      switch (tensor.value()->storage_type()) {
        case MemoryStorageType::kDevice:
          continue;
        case MemoryStorageType::kHost:
        case MemoryStorageType::kSystem:
          operation = cudaMemcpyHostToDevice;
          break;
        default:
          GXF_LOG_ERROR("Unsupported MemoryStorageType in tensor");
          return GXF_FAILURE;
      }
      storage_type = MemoryStorageType::kDevice;
    } else if (CopyMode(mode_.get()) == CopyMode::kCopyToHost) {
      switch (tensor.value()->storage_type()) {
        case MemoryStorageType::kHost:
          continue;
        case MemoryStorageType::kDevice:
          operation = cudaMemcpyDeviceToHost;
          break;
        case MemoryStorageType::kSystem:
          operation = cudaMemcpyHostToHost;
          break;
        default:
          GXF_LOG_ERROR("Unsupported MemoryStorageType in tensor");
          return GXF_FAILURE;
      }
      storage_type = MemoryStorageType::kHost;
    } else if (CopyMode(mode_.get()) == CopyMode::kCopyToSystem) {
      switch (tensor.value()->storage_type()) {
        case MemoryStorageType::kSystem:
          continue;
        case MemoryStorageType::kDevice:
          operation = cudaMemcpyDeviceToHost;
          break;
        case MemoryStorageType::kHost:
          operation = cudaMemcpyHostToHost;
          break;
        default:
          GXF_LOG_ERROR("Unsupported MemoryStorageType in tensor");
          return GXF_FAILURE;
      }
      storage_type = MemoryStorageType::kSystem;
    } else {
      GXF_LOG_ERROR("Unsupported value for 'copy-mode' parameter");
      return GXF_FAILURE;
    }

    const Tensor temp = std::move(*tensor.value());

    Tensor::stride_array_t strides;
    for (size_t i = 0; i < strides.size(); i++) {
      strides[i] = temp.stride(i);
    }

    Expected<void> result = tensor.value()->reshapeCustom(temp.shape(), temp.element_type(),
                                                          temp.bytes_per_element(),
                                                          strides, storage_type, allocator_);
    if (!result) {
      return ToResultCode(result);
    }

    cudaError_t error = cudaMemcpy(tensor.value()->pointer(), temp.pointer(),
                                   tensor.value()->size(), operation);
    if (error != cudaSuccess) {
      GXF_LOG_ERROR("Failure in CudaMemcpy Error: %s", cudaGetErrorString(error));
      return GXF_FAILURE;
    }
  }

  return ToResultCode(transmitter_->publish(entity.value()));
}

}  // namespace gxf
}  // namespace nvidia
