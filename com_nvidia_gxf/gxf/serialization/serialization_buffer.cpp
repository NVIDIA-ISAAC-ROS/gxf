/*
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/serialization/serialization_buffer.hpp"

#include <cstring>

namespace nvidia {
namespace gxf {

gxf_result_t SerializationBuffer::registerInterface(Registrar* registrar) {
  if (registrar == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  Expected<void> result;
  result &= registrar->parameter(
      allocator_, "allocator", "Allocator",
      "Memory allocator");
  result &= registrar->parameter(
      buffer_size_, "buffer_size", "Buffer Size",
      "Initial size of the buffer in bytes (4kB by default)",
      static_cast<size_t>(1 << 12));
  result &= registrar->parameter(
      storage_type_, "storage_type", "Storage type",
      "The initial memory storage type used by this buffer (kSystem by default)", 2);
  return ToResultCode(result);
}

gxf_result_t SerializationBuffer::initialize() {
  return ToResultCode(resize(buffer_size_, static_cast<MemoryStorageType>(storage_type_.get())));
}

gxf_result_t SerializationBuffer::write_abi(const void* data, size_t size, size_t* bytes_written) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (data == nullptr || bytes_written == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  if (size > buffer_.size() - write_offset_) {
    return GXF_EXCEEDING_PREALLOCATED_SIZE;
  }
  std::memcpy(buffer_.pointer() + write_offset_, data, size);
  write_offset_ += size;
  *bytes_written = size;
  return GXF_SUCCESS;
}

gxf_result_t SerializationBuffer::read_abi(void* data, size_t size, size_t* bytes_read) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (data == nullptr || bytes_read == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  if (size > buffer_.size() - read_offset_) {
    return GXF_EXCEEDING_PREALLOCATED_SIZE;
  }
  std::memcpy(data, buffer_.pointer() + read_offset_, size);
  read_offset_ += size;
  *bytes_read = size;
  return GXF_SUCCESS;
}

Expected<void> SerializationBuffer::resize(size_t size, MemoryStorageType storage_type) {
  std::unique_lock<std::mutex> lock(mutex_);
  write_offset_ = 0;
  read_offset_ = 0;
  return buffer_.resize(allocator_, size, storage_type);
}

Expected<void> SerializationBuffer::wrapMemory(void* pointer, uint64_t size,
                                               MemoryStorageType storage_type,
                                               release_function_t release_func) {
  std::unique_lock<std::mutex> lock(mutex_);
  write_offset_ = 0;
  read_offset_ = 0;
  return buffer_.wrapMemory(pointer, size, storage_type, release_func);
}

size_t SerializationBuffer::size() const {
  std::unique_lock<std::mutex> lock(mutex_);
  return write_offset_;
}

void SerializationBuffer::reset() {
  std::unique_lock<std::mutex> lock(mutex_);
  write_offset_ = 0;
  read_offset_ = 0;
}

}  // namespace gxf
}  // namespace nvidia
