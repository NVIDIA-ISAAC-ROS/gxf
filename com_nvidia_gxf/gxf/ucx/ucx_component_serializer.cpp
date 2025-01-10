/*
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "ucx_component_serializer.hpp"

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

#pragma pack(push, 1)
struct ColorPlaneHeader {
  char color_space[256];
  uint8_t bytes_per_pixel;
  int32_t stride;
  uint32_t offset = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint64_t size = 0;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct VideoBufferHeader {
  uint32_t width;                     // width of a video frame
  uint32_t height;                    // height of a video frame
  MemoryStorageType storage_type;     // CPU or GPU tensor
  VideoFormat color_format;           // Color format of a video frame
  ColorPlaneHeader color_planes[4];   // Information for single color plane
  SurfaceLayout surface_layout;       // surface memory layout of a video frame
  uint32_t number_of_color_planes;    // number of color planes
};

#pragma pack(pop)

#pragma pack(push, 1)
struct AudioBufferHeader {
  MemoryStorageType storage_type;     // CPU or GPU tensor
  AudioBufferInfo buffer_info;
};
#pragma pack(pop)

}  // namespace

gxf_result_t UcxComponentSerializer::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
    allocator_, "allocator", "Memory allocator",
    "Memory allocator for tensor components");
  return ToResultCode(result);
}

gxf_result_t UcxComponentSerializer::initialize() {
  if (!IsLittleEndian()) {
    GXF_LOG_WARNING("UcxComponentSerializer currently only supports little-endian devices");
    return GXF_NOT_IMPLEMENTED;
  }
  return ToResultCode(configureSerializers() & configureDeserializers());
}

Expected<void> UcxComponentSerializer::configureSerializers() {
  Expected<void> result;
  result &= setSerializer<Timestamp>(
    [this](void* component, Endpoint* endpoint) {
      return serializeTimestamp(*static_cast<Timestamp*>(component), endpoint);
    });
  result &= setSerializer<Tensor>(
    [this](void* component, Endpoint* endpoint) {
      return serializeTensor(*static_cast<Tensor*>(component), endpoint);
    });
  result &= setSerializer<VideoBuffer>(
    [this](void* component, Endpoint* endpoint) {
      return serializeVideoBuffer(*static_cast<VideoBuffer*>(component), endpoint);
    });
  result &= setSerializer<AudioBuffer>(
    [this](void* component, Endpoint* endpoint) {
      return serializeAudioBuffer(*static_cast<AudioBuffer*>(component), endpoint);
    });
  result &= setSerializer<EndOfStream>(
    [this](void* component, Endpoint* endpoint) {
      return serializeEndOfStream(*static_cast<EndOfStream*>(component), endpoint);
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

Expected<void> UcxComponentSerializer::configureDeserializers() {
  Expected<void> result;
  result &= setDeserializer<Timestamp>(
    [this](void* component, Endpoint* endpoint) {
      return deserializeTimestamp(endpoint).assign_to(*static_cast<Timestamp*>(component));
    });
  result &= setDeserializer<Tensor>(
    [this](void* component, Endpoint* endpoint) {
      return deserializeTensor(endpoint).assign_to(*static_cast<Tensor*>(component));
    });
  result &= setDeserializer<VideoBuffer>(
    [this](void* component, Endpoint* endpoint) {
      return deserializeVideoBuffer(endpoint).assign_to(*static_cast<VideoBuffer*>(component));
    });
  result &= setDeserializer<AudioBuffer>(
    [this](void* component, Endpoint* endpoint) {
      return deserializeAudioBuffer(endpoint).assign_to(*static_cast<AudioBuffer*>(component));
    });
  result &= setDeserializer<EndOfStream>(
    [this](void* component, Endpoint* endpoint) {
      return deserializeEndOfStream(endpoint).assign_to(*static_cast<EndOfStream*>(component));
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

Expected<size_t> UcxComponentSerializer::serializeTimestamp(Timestamp timestamp,
                                                            Endpoint* endpoint) {
  if (!endpoint) { return Unexpected{GXF_ARGUMENT_NULL}; }
  return endpoint->writeTrivialType<Timestamp>(&timestamp);
}

Expected<Timestamp> UcxComponentSerializer::deserializeTimestamp(Endpoint* endpoint) {
  if (!endpoint) { return Unexpected{GXF_ARGUMENT_NULL}; }
  Timestamp timestamp;
  return ExpectedOrError(endpoint->readTrivialType<Timestamp>(&timestamp), timestamp);
}

Expected<size_t> UcxComponentSerializer::serializeTensor(const Tensor& tensor, Endpoint* endpoint) {
  TensorHeader header;
  header.storage_type = tensor.storage_type();
  header.element_type = tensor.element_type();
  header.bytes_per_element = tensor.bytes_per_element();
  header.rank = tensor.rank();
  for (size_t i = 0; i < Shape::kMaxRank; i++) {
    header.dims[i] = tensor.shape().dimension(i);
    header.strides[i] = tensor.stride(i);
  }
  auto result = endpoint->write_ptr(tensor.pointer(), tensor.size(), tensor.storage_type());
  if (!result) {
    return ForwardError(result);
  }
  auto size = endpoint->writeTrivialType<TensorHeader>(&header);
  if (!size) {
    return ForwardError(size);
  }
  return sizeof(header);
}

Expected<Tensor> UcxComponentSerializer::deserializeTensor(Endpoint* endpoint) {
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
  result = endpoint->write_ptr(tensor.pointer(), tensor.size(), tensor.storage_type());
  if (!result) {
    return ForwardError(result);
  }
  return tensor;
}

Expected<size_t>
UcxComponentSerializer::serializeVideoBuffer(const VideoBuffer& videoBuffer, Endpoint* endpoint) {
  VideoBufferHeader header;
  header.storage_type = videoBuffer.storage_type();
  header.width = videoBuffer.video_frame_info().width;
  header.height = videoBuffer.video_frame_info().height;
  header.color_format = videoBuffer.video_frame_info().color_format;
  header.surface_layout = videoBuffer.video_frame_info().surface_layout;
  header.number_of_color_planes = videoBuffer.video_frame_info().color_planes.size();

  for (size_t i = 0; i < header.number_of_color_planes; i++) {
    strncpy(header.color_planes[i].color_space,
            videoBuffer.video_frame_info().color_planes[i].color_space.c_str(),
            videoBuffer.video_frame_info().color_planes[i].color_space.size());
    header.color_planes[i].height = videoBuffer.video_frame_info().color_planes[i].height;
    header.color_planes[i].width = videoBuffer.video_frame_info().color_planes[i].width;
    header.color_planes[i].offset = videoBuffer.video_frame_info().color_planes[i].offset;
    header.color_planes[i].size = videoBuffer.video_frame_info().color_planes[i].size;
    header.color_planes[i].stride = videoBuffer.video_frame_info().color_planes[i].stride;
    header.color_planes[i].bytes_per_pixel = videoBuffer.video_frame_info().
                                             color_planes[i].bytes_per_pixel;
  }

  auto result = endpoint->write_ptr(videoBuffer.pointer(), videoBuffer.size(),
                                    videoBuffer.storage_type());
  if (!result) {
    return ForwardError(result);
  }

  auto size = endpoint->writeTrivialType<VideoBufferHeader>(&header);
  if (!size) {
    return ForwardError(size);
  }
  return sizeof(header);
}

Expected<VideoBuffer> UcxComponentSerializer::deserializeVideoBuffer(Endpoint* endpoint) {
  if (!endpoint) {
    return Unexpected{GXF_ARGUMENT_NULL};
  }

  VideoBufferHeader header;
  auto size = endpoint->readTrivialType<VideoBufferHeader>(&header);
  if (!size) {
    return ForwardError(size);
  }
  VideoBufferInfo buffer_info;
  buffer_info.color_format = header.color_format;
  buffer_info.height = header.height;
  buffer_info.width = header.width;
  buffer_info.surface_layout = header.surface_layout;

  uint64_t video_buffer_size = 0;
  for (size_t i = 0; i < header.number_of_color_planes; i++) {
    ColorPlane color_plane;
    color_plane.bytes_per_pixel = header.color_planes[i].bytes_per_pixel;
    color_plane.color_space = header.color_planes[i].color_space;
    color_plane.height = header.color_planes[i].height;
    color_plane.offset = header.color_planes[i].offset;
    color_plane.size = header.color_planes[i].size;
    color_plane.stride = header.color_planes[i].stride;
    color_plane.width = header.color_planes[i].width;
    video_buffer_size += color_plane.size;
    buffer_info.color_planes.push_back(color_plane);
  }

  VideoBuffer videoBuffer;
  auto result = videoBuffer.resizeCustom(buffer_info, video_buffer_size,
                                         header.storage_type, allocator_);
  if (!result) {
    return ForwardError(result);
  }
  result = endpoint->write_ptr(videoBuffer.pointer(), videoBuffer.size(),
                               videoBuffer.storage_type());
  if (!result) {
    return ForwardError(result);
  }
  return videoBuffer;
}

Expected<size_t>
UcxComponentSerializer::serializeAudioBuffer(const AudioBuffer& audioBuffer, Endpoint* endpoint) {
  AudioBufferHeader header;
  header.storage_type = audioBuffer.storage_type();
  header.buffer_info = audioBuffer.audio_buffer_info();
  auto result = endpoint->write_ptr(audioBuffer.pointer(), audioBuffer.size(),
                                    audioBuffer.storage_type());
  if (!result) {
    return ForwardError(result);
  }
  auto size = endpoint->writeTrivialType<AudioBufferHeader>(&header);
  if (!size) {
    return ForwardError(size);
  }
  return sizeof(header);
}

Expected<AudioBuffer> UcxComponentSerializer::deserializeAudioBuffer(Endpoint* endpoint) {
  if (!endpoint) {
    return Unexpected{GXF_ARGUMENT_NULL};
  }

  AudioBufferHeader header;
  auto size = endpoint->readTrivialType<AudioBufferHeader>(&header);
  if (!size) {
    return ForwardError(size);
  }
  AudioBufferInfo buffer_info = header.buffer_info;

  AudioBuffer audioBuffer;
  auto result = audioBuffer.resizeCustom(buffer_info, header.storage_type, allocator_);
  if (!result) {
    return ForwardError(result);
  }
  result = endpoint->write_ptr(audioBuffer.pointer(), audioBuffer.size(),
                               audioBuffer.storage_type());
  if (!result) {
    return ForwardError(result);
  }
  return audioBuffer;
}

Expected<size_t> UcxComponentSerializer::serializeEndOfStream(EndOfStream& eos,
                                                              Endpoint* endpoint) {
  return serializeInteger<int64_t>(eos.stream_id(), endpoint);
}

Expected<EndOfStream> UcxComponentSerializer::deserializeEndOfStream(Endpoint* endpoint) {
  if (!endpoint) {
    return Unexpected{GXF_ARGUMENT_NULL};
  }

  EndOfStream eos;
  eos.stream_id(deserializeInteger<int64_t>(endpoint).value());

  return eos;
}

}  // namespace gxf
}  // namespace nvidia
