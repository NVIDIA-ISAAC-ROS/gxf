/*
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/serialization/std_component_serializer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <utility>

namespace nvidia {
namespace gxf {

namespace {

#pragma pack(push, 1)
struct TensorHeader {
  MemoryStorageType storage_type;     // CPU or GPU tensor
  PrimitiveType element_type;         // Tensor element type
  uint64_t bytes_per_element;         // Bytes per tensor element
  uint32_t rank;                      // Tensor rank
  int32_t dims[Shape::kMaxRank];      // Tensor dimensions
  uint64_t strides[Shape::kMaxRank];  // Tensor strides
};
#pragma pack(pop)

}  // namespace

gxf_result_t StdComponentSerializer::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
    allocator_, "allocator", "Memory allocator",
    "Memory allocator for tensor components");
  return ToResultCode(result);
}

gxf_result_t StdComponentSerializer::initialize() {
  if (!IsLittleEndian()) {
    GXF_LOG_WARNING("StdComponentSerializer currently only supports little-endian devices");
    return GXF_NOT_IMPLEMENTED;
  }
  return ToResultCode(configureSerializers() & configureDeserializers());
}

Expected<void> StdComponentSerializer::configureSerializers() {
  Expected<void> result;
  result &= setSerializer<Timestamp>(
    [this](void* component, Endpoint* endpoint) {
      return serializeTimestamp(*static_cast<Timestamp*>(component), endpoint);
    });
  result &= setSerializer<Tensor>(
    [this](void* component, Endpoint* endpoint) {
      return serializeTensor(*static_cast<Tensor*>(component), endpoint);
    });
  result &= setSerializer<int8_t>(
    [this](void* component, Endpoint* endpoint) {
      return serializeInteger<int8_t>(*static_cast<int8_t*>(component), endpoint);
    });
  result &= setSerializer<uint8_t>(
    [this](void* component, Endpoint* endpoint) {
      return serializeInteger<uint8_t>(*static_cast<uint8_t*>(component), endpoint);
    });
  result &= setSerializer<int16_t>(
    [this](void* component, Endpoint* endpoint) {
      return serializeInteger<int16_t>(*static_cast<int16_t*>(component), endpoint);
    });
  result &= setSerializer<uint16_t>(
    [this](void* component, Endpoint* endpoint) {
      return serializeInteger<uint16_t>(*static_cast<uint16_t*>(component), endpoint);
    });
  result &= setSerializer<int32_t>(
    [this](void* component, Endpoint* endpoint) {
      return serializeInteger<int32_t>(*static_cast<int32_t*>(component), endpoint);
    });
  result &= setSerializer<uint32_t>(
    [this](void* component, Endpoint* endpoint) {
      return serializeInteger<uint32_t>(*static_cast<uint32_t*>(component), endpoint);
    });
  result &= setSerializer<int64_t>(
    [this](void* component, Endpoint* endpoint) {
      return serializeInteger<int64_t>(*static_cast<int64_t*>(component), endpoint);
    });
  result &= setSerializer<uint64_t>(
    [this](void* component, Endpoint* endpoint) {
      return serializeInteger<uint64_t>(*static_cast<uint64_t*>(component), endpoint);
    });
  result &= setSerializer<float>(
    [this](void* component, Endpoint* endpoint) {
      return endpoint->writeTrivialType(static_cast<float*>(component));
    });
  result &= setSerializer<double>(
    [this](void* component, Endpoint* endpoint) {
      return endpoint->writeTrivialType(static_cast<double*>(component));
    });
  result &= setSerializer<bool>(
    [this](void* component, Endpoint* endpoint) {
      return endpoint->writeTrivialType(static_cast<bool*>(component));
    });
  return result;
}

Expected<void> StdComponentSerializer::configureDeserializers() {
  Expected<void> result;
  result &= setDeserializer<Timestamp>(
    [this](void* component, Endpoint* endpoint) {
      return deserializeTimestamp(endpoint).assign_to(*static_cast<Timestamp*>(component));
    });
  result &= setDeserializer<Tensor>(
    [this](void* component, Endpoint* endpoint) {
      return deserializeTensor(endpoint).assign_to(*static_cast<Tensor*>(component));
    });
  result &= setDeserializer<int8_t>(
    [this](void* component, Endpoint* endpoint) {
      return deserializeInteger<int8_t>(endpoint).assign_to(*static_cast<int8_t*>(component));
    });
  result &= setDeserializer<uint8_t>(
    [this](void* component, Endpoint* endpoint) {
      return deserializeInteger<uint8_t>(endpoint).assign_to(*static_cast<uint8_t*>(component));
    });
  result &= setDeserializer<int16_t>(
    [this](void* component, Endpoint* endpoint) {
      return deserializeInteger<int16_t>(endpoint).assign_to(*static_cast<int16_t*>(component));
    });
  result &= setDeserializer<uint16_t>(
    [this](void* component, Endpoint* endpoint) {
      return deserializeInteger<uint16_t>(endpoint).assign_to(*static_cast<uint16_t*>(component));
    });
  result &= setDeserializer<int32_t>(
    [this](void* component, Endpoint* endpoint) {
      return deserializeInteger<int32_t>(endpoint).assign_to(*static_cast<int32_t*>(component));
    });
  result &= setDeserializer<uint32_t>(
    [this](void* component, Endpoint* endpoint) {
      return deserializeInteger<uint32_t>(endpoint).assign_to(*static_cast<uint32_t*>(component));
    });
  result &= setDeserializer<int64_t>(
    [this](void* component, Endpoint* endpoint) {
      return deserializeInteger<int64_t>(endpoint).assign_to(*static_cast<int64_t*>(component));
    });
  result &= setDeserializer<uint64_t>(
    [this](void* component, Endpoint* endpoint) {
      return deserializeInteger<uint64_t>(endpoint).assign_to(*static_cast<uint64_t*>(component));
    });
  result &= setDeserializer<float>(
    [this](void* component, Endpoint* endpoint) {
      return endpoint->readTrivialType(static_cast<float*>(component)) & Success;
    });
  result &= setDeserializer<double>(
    [this](void* component, Endpoint* endpoint) {
      return endpoint->readTrivialType(static_cast<double*>(component)) & Success;
    });
  result &= setDeserializer<bool>(
    [this](void* component, Endpoint* endpoint) {
      return endpoint->readTrivialType(static_cast<bool*>(component)) & Success;
    });
  return result;
}

Expected<size_t> StdComponentSerializer::serializeTimestamp(Timestamp timestamp,
                                                            Endpoint* endpoint) {
  if (!endpoint) { return Unexpected{GXF_ARGUMENT_NULL}; }
  return endpoint->writeTrivialType<Timestamp>(&timestamp);
}

Expected<Timestamp> StdComponentSerializer::deserializeTimestamp(Endpoint* endpoint) {
  if (!endpoint) { return Unexpected{GXF_ARGUMENT_NULL}; }
  Timestamp timestamp;
  return ExpectedOrError(endpoint->readTrivialType<Timestamp>(&timestamp), timestamp);
}

Expected<size_t> StdComponentSerializer::serializeTensor(const Tensor& tensor, Endpoint* endpoint) {
  if (!endpoint) {
    return Unexpected{GXF_ARGUMENT_NULL};
  }

  TensorHeader header;
  header.storage_type = tensor.storage_type();
  header.element_type = tensor.element_type();
  header.bytes_per_element = tensor.bytes_per_element();
  header.rank = tensor.rank();
  for (size_t i = 0; i < Shape::kMaxRank; i++) {
    header.dims[i] = tensor.shape().dimension(i);
    header.strides[i] = tensor.stride(i);
  }

  auto size = endpoint->writeTrivialType<TensorHeader>(&header);
  if (!size) {
    return ForwardError(size);
  }

  const size_t tensor_size = tensor.element_count() * tensor.bytes_per_element();

  switch (tensor.storage_type()) {
    case MemoryStorageType::kHost:
    case MemoryStorageType::kSystem:
      {
        auto size = endpoint->write(tensor.pointer(), tensor_size);
        if (!size) {
          return ForwardError(size);
        }
      }
      break;
    case MemoryStorageType::kDevice:
      {
        auto buffer = allocator_->allocate(tensor_size, MemoryStorageType::kHost);
        if (!buffer) {
          return ForwardError(buffer);
        }
        const cudaError_t error = cudaMemcpy(buffer.value(), tensor.pointer(), tensor_size,
                                             cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
        GXF_LOG_ERROR("Failure in CudaMemcpy. cuda_error: %s, error_str: %s",
                      cudaGetErrorName(error), cudaGetErrorString(error));
          return Unexpected{GXF_FAILURE};
        }
        auto size = endpoint->write(buffer.value(), tensor_size);
        if (!size) {
          return ForwardError(size);
        }
        auto result = allocator_->free(buffer.value());
        if (!result) {
          return ForwardError(result);
        }
      }
      break;
    default:
      GXF_LOG_ERROR("Invalid memory storage type %d specified for tensor storage",
      static_cast<int>(tensor.storage_type()));
      return Unexpected{GXF_FAILURE};
  }

  return sizeof(header) + tensor_size;
}

Expected<Tensor> StdComponentSerializer::deserializeTensor(Endpoint* endpoint) {
  if (!endpoint) {
    return Unexpected{GXF_ARGUMENT_NULL};
  }

  TensorHeader header;
  auto size = endpoint->readTrivialType<TensorHeader>(&header);
  if (!size) {
    return ForwardError(size);
  }

  std::array<int32_t, Shape::kMaxRank> dims;
  if (sizeof(header.dims) > Shape::kMaxRank * sizeof(int32_t)) {
        GXF_LOG_ERROR("Header size exceeds limit of %lu.",
                      Shape::kMaxRank * sizeof(int32_t));
          return Unexpected{GXF_FAILURE};
  }

  std::memcpy(dims.data(), header.dims, sizeof(header.dims));
  Tensor::stride_array_t strides;
  if (sizeof(header.strides) > Shape::kMaxRank * sizeof(int64_t)) {
        GXF_LOG_ERROR("Header size exceeds limit of %lu.",
                      Shape::kMaxRank * sizeof(int64_t));
          return Unexpected{GXF_FAILURE};
  }

  std::memcpy(strides.data(), header.strides, sizeof(header.strides));
  Tensor tensor;
  auto result = tensor.reshapeCustom(Shape(dims, header.rank),
                                     header.element_type, header.bytes_per_element, strides,
                                     header.storage_type, allocator_);
  if (!result) {
    return ForwardError(result);
  }

  const size_t tensor_size = tensor.element_count() * tensor.bytes_per_element();

  switch (tensor.storage_type()) {
    case MemoryStorageType::kHost:
    case MemoryStorageType::kSystem:
      {
        auto size = endpoint->read(tensor.pointer(), tensor_size);
        if (!size) {
          return ForwardError(size);
        }
      }
      break;
    case MemoryStorageType::kDevice:
      {
        auto buffer = allocator_->allocate(tensor_size, MemoryStorageType::kHost);
        if (!buffer) {
          return ForwardError(buffer);
        }
        auto size = endpoint->read(buffer.value(), tensor_size);
        if (!size) {
          return ForwardError(size);
        }
        const cudaError_t error = cudaMemcpy(tensor.pointer(), buffer.value(), tensor_size,
                                             cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
          GXF_LOG_ERROR("Failure in CudaMemcpy. cuda_error: %s, error_str: %s",
                        cudaGetErrorName(error), cudaGetErrorString(error));
          return Unexpected{GXF_FAILURE};
        }
        auto result = allocator_->free(buffer.value());
        if (!result) {
          return ForwardError(result);
        }
      }
      break;
    default:
      return Unexpected{GXF_FAILURE};
  }

  return tensor;
}

}  // namespace gxf
}  // namespace nvidia
