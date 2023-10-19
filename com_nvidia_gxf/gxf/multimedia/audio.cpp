/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <type_traits>

#include "gxf/multimedia/audio.hpp"

namespace nvidia {
namespace gxf {

Expected<void> AudioBuffer::resizeCustom(AudioBufferInfo buffer_info,
                                         MemoryStorageType storage_type,
                                         Handle<Allocator> allocator) {
  if (!allocator) { return Unexpected{GXF_ARGUMENT_NULL}; }

  if ((buffer_info.audio_layout == AudioLayout::GXF_AUDIO_LAYOUT_CUSTOM) ||
      (buffer_info.audio_format == AudioFormat::GXF_AUDIO_FORMAT_CUSTOM)) {
    return Unexpected{GXF_ARGUMENT_INVALID};
  }

  buffer_info_ = buffer_info;
  uint64_t size = buffer_info_.bytes_per_sample * buffer_info.samples * buffer_info.channels;

  auto result = memory_buffer_.freeBuffer();
  if (!result) { return ForwardError(result); }

  result = memory_buffer_.resize(allocator, size, storage_type);
  if (!result) { return ForwardError(result); }

  return Success;
}

Expected<void> AudioBuffer::wrapMemory(AudioBufferInfo buffer_info, uint64_t size,
                                      MemoryStorageType storage_type,
                                      void* pointer,
                                      release_function_t release_func) {
  auto result = memory_buffer_.freeBuffer();
  if (!result) { return ForwardError(result); }

  buffer_info_ = buffer_info;

  result = memory_buffer_.wrapMemory(pointer, size, storage_type, release_func);
  if (!result) { return ForwardError(result); }

  return Success;
}

}  // namespace gxf
}  // namespace nvidia
