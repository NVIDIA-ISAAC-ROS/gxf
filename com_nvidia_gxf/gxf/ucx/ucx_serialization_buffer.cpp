/*
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "ucx_common.hpp"
#include "ucx_serialization_buffer.hpp"

#include <cstring>

namespace nvidia {
namespace gxf {

gxf_result_t UcxSerializationBuffer::registerInterface(Registrar* registrar) {
  if (registrar == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  Expected<void> result;
  result &= registrar->parameter(
      allocator_, "allocator", "Allocator",
      "Memory allocator");
  result &= registrar->parameter(
      buffer_size_, "buffer_size", "Buffer Size",
      "Size of the buffer in bytes (4kB by default)",
      static_cast<size_t>(1 << 12));
  return ToResultCode(result);
}

gxf_result_t UcxSerializationBuffer::initialize() {
  write_offset_ = 0;
  read_offset_ = 0;
  return ToResultCode(buffer_.resize(allocator_, buffer_size_, MemoryStorageType::kSystem));
}

Expected<void> UcxSerializationBuffer::set_allocator(Handle<Allocator> allocator) {
  if (allocator.is_null()) { return Unexpected{GXF_ARGUMENT_NULL}; }
  return allocator_.set(allocator);
}

gxf_result_t UcxSerializationBuffer::write_abi(const void* data, size_t size,
                                               size_t* bytes_written) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (data == nullptr || bytes_written == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  if (buffer_.size() == 0) {
    return GXF_UNINITIALIZED_VALUE;
  }
  if (size > buffer_.size() - write_offset_) {
    return GXF_EXCEEDING_PREALLOCATED_SIZE;
  }
  std::memcpy(buffer_.pointer() + write_offset_, data, size);
  write_offset_ += size;
  *bytes_written = size;
  return GXF_SUCCESS;
}

gxf_result_t UcxSerializationBuffer::read_abi(void* data, size_t size, size_t* bytes_read) {
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

gxf_result_t UcxSerializationBuffer::write_ptr_abi(const void* pointer, size_t size,
                                                   MemoryStorageType type) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (pointer == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  ucp_dt_iov_t iov;
  iov.buffer = const_cast<void*>(pointer);
  iov.length = size;
  iov_buffers_.push_back(iov);
  mem_type_ = ucx_mem_type(type);
  return GXF_SUCCESS;
}

gxf::Expected<void> UcxSerializationBuffer::resize(size_t size) {
  std::unique_lock<std::mutex> lock(mutex_);
  write_offset_ = 0;
  read_offset_ = 0;
  return buffer_size_.set(size)
      .and_then([&]() {
        return buffer_.resize(allocator_, buffer_size_, MemoryStorageType::kSystem);
      });
}

size_t UcxSerializationBuffer::size() const {
  std::unique_lock<std::mutex> lock(mutex_);
  return write_offset_;
}

void UcxSerializationBuffer::reset() {
  std::unique_lock<std::mutex> lock(mutex_);
  write_offset_ = 0;
  read_offset_ = 0;
  iov_buffers_.clear();
  mem_type_ = UCS_MEMORY_TYPE_HOST;
}

}  // namespace gxf
}  // namespace nvidia
