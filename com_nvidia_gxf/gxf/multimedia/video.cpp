/*
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <cstdint>
#include <type_traits>
#include <utility>

#include "gxf/multimedia/video.hpp"

namespace nvidia {
namespace gxf {

Expected<void> VideoBuffer::resizeCustom(VideoBufferInfo buffer_info, uint64_t size,
                                         MemoryStorageType storage_type,
                                         Handle<Allocator> allocator) {
  if (!allocator) { return Unexpected{GXF_ARGUMENT_NULL}; }

  if ((buffer_info.color_format == VideoFormat::GXF_VIDEO_FORMAT_CUSTOM) ||
      (buffer_info.width == 0) || (buffer_info.height == 0)) {
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  buffer_info_ = buffer_info;

  auto result = memory_buffer_.freeBuffer();
  if (!result) { return ForwardError(result); }

  result = memory_buffer_.resize(allocator, size, storage_type);
  if (!result) { return ForwardError(result); }

  return Success;
}

Expected<void> VideoBuffer::wrapMemory(VideoBufferInfo buffer_info, uint64_t size,
                                       MemoryStorageType storage_type, void* pointer,
                                       release_function_t release_func) {
  auto result = memory_buffer_.freeBuffer();
  if (!result) { return ForwardError(result); }

  buffer_info_ = buffer_info;

  result = memory_buffer_.wrapMemory(pointer, size, storage_type, release_func);
  if (!result) { return ForwardError(result); }

  return Success;
}

Expected<void> VideoBuffer::moveToTensor(Tensor* tensor) {
  if (!tensor) {
    GXF_LOG_ERROR("VideoBuffer received invalid tensor pointer");
    return Unexpected{GXF_ARGUMENT_NULL};
  }

  Expected<PrimitiveType> primitive_type = getPlanarPrimitiveType(buffer_info_.color_format);
  if (!primitive_type) { return ForwardError(primitive_type); }

  auto c = static_cast<int32_t>(buffer_info_.color_planes.size());
  auto h = static_cast<int32_t>(buffer_info_.height);
  auto w = static_cast<int32_t>(buffer_info_.width);

  if ((c < 1) || (h < 1) || (w < 1)) {
    GXF_LOG_ERROR(
        "VideoBuffer cannot be converted to tensor."
        " Invalid dimensions [CHW]:[%d,%d,%d]",
        c, h, w);
    return Unexpected{GXF_INVALID_DATA_FORMAT};
  }

  // Ignore channel dims of 1
  // 1xnxm - single plane image (rank = 2)
  // pxnxm - multi plane image (rank = 3)
  Shape tensor_shape;
  if (c == 1) {
    tensor_shape = Shape({w, h});
  } else {
    tensor_shape = Shape({w, h, c});
  }

  uint64_t bytes_per_pixel = 0;
  Tensor::stride_array_t tensor_strides;
  for (size_t i = 0; i < buffer_info_.color_planes.size(); ++i) {
    bytes_per_pixel += buffer_info_.color_planes[i].bytes_per_pixel;
    tensor_strides[i] = buffer_info_.color_planes[i].stride;
  }

  auto result = tensor->wrapMemoryBuffer(tensor_shape, primitive_type.value(), bytes_per_pixel,
                                         tensor_strides, std::move(memory_buffer_));
  return result;
}

Expected<void> VideoBuffer::moveToTensor(Handle<Tensor>& tensor) {
  if (!tensor) {
    GXF_LOG_ERROR("VideoBuffer received invalid tensor handle");
    return Unexpected{GXF_ARGUMENT_NULL};
  }
  auto result = tensor.try_get();
  if (!result) {
    GXF_LOG_ERROR("VideoBuffer received invalid tensor handle");
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  return moveToTensor(result.value());
}

}  // namespace gxf
}  // namespace nvidia
