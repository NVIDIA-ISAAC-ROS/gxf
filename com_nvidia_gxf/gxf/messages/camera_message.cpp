/*
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <array>
#include <vector>

#include "camera_message.hpp"
#include "gxf/std/gems/video_buffer/allocator.hpp"

namespace nvidia {
namespace gxf {
namespace {
constexpr const char kNameFrame[] = "frame";
constexpr const char kNameIntrinsics[] = "intrinsics";
constexpr const char kNameExtrinsics[] = "extrinsics";
constexpr const char kNameSequenceNumber[] = "sequence_number";
// Adds CameraMessage components to an entity
gxf::Expected<CameraMessageParts> InitializeCameraMessage(gxf_context_t context) {
  CameraMessageParts message;
  return gxf::Entity::New(context)
      .assign_to(message.entity)
      .and_then([&]() {
        return message.entity.add<gxf::VideoBuffer>(kNameFrame);
      })
      .assign_to(message.frame)
      .log_error("Failed to add '%s' to message entity", kNameFrame)
      .and_then([&]() {
        return message.entity.add<gxf::CameraModel>(kNameIntrinsics);
      })
      .assign_to(message.intrinsics)
      .log_error("Failed to add '%s' to message entity", kNameIntrinsics)
      .and_then([&]() {
        return message.entity.add<gxf::Pose3D>(kNameExtrinsics);
      })
      .assign_to(message.extrinsics)
      .log_error("Failed to add '%s' to message entity", kNameExtrinsics)
      .and_then([&]() {
        return message.entity.add<int64_t>(kNameSequenceNumber);
      })
      .assign_to(message.sequence_number)
      .log_error("Failed to add '%s' to message entity", kNameSequenceNumber)
      .and_then([&]() {
        return message.entity.add<gxf::Timestamp>();
      })
      .assign_to(message.timestamp)
      .log_error("Failed to add timestamp to message entity %ld", message.entity.eid())
      .substitute(message);
}
}  // namespace
template <gxf::VideoFormat Format>
gxf::Expected<CameraMessageParts> CreateCameraMessage(gxf_context_t context,
                                                      uint32_t width,
                                                      uint32_t height,
                                                      gxf::SurfaceLayout layout,
                                                      gxf::MemoryStorageType storage_type,
                                                      gxf::Handle<gxf::Allocator> allocator,
                                                      bool padded) {
  auto message = InitializeCameraMessage(context);
  if (!message) {
    return gxf::ForwardError(message);
  }
  if (padded) {
    auto result = message->frame->resize<Format>(width, height, layout, storage_type, allocator);
    if (!result) {
      return gxf::ForwardError(result);
    }
  } else {
    auto result = AllocateUnpaddedVideoBuffer<Format>(
        message->frame, width, height, storage_type, allocator);
    if (!result) {
      return gxf::ForwardError(result);
    }
  }
  return message;
}
gxf::Expected<CameraMessageParts> CreateCameraMessage(gxf_context_t context,
                                                      gxf::VideoBufferInfo buffer_info,
                                                      uint64_t size,
                                                      gxf::MemoryStorageType storage_type,
                                                      gxf::Handle<gxf::Allocator> allocator) {
  auto message = InitializeCameraMessage(context);
  if (!message) {
    return gxf::ForwardError(message);
  }
  auto result = message->frame->resizeCustom(buffer_info, size, storage_type, allocator);
  if (!result) {
    return gxf::ForwardError(result);
  }
  return message;
}
gxf::Expected<CameraMessageParts> GetCameraMessage(const gxf::Entity entity) {
  CameraMessageParts message;
  message.entity = entity;
  return message.entity.get<gxf::VideoBuffer>(kNameFrame)
      .assign_to(message.frame)
      .log_error("Failed to get '%s' from message entity", kNameFrame)
      .and_then([&]() {
        return message.entity.get<gxf::CameraModel>(kNameIntrinsics);
      })
      .assign_to(message.intrinsics)
      .log_error("Failed to get '%s' from message entity", kNameIntrinsics)
      .and_then([&]() {
        return message.entity.get<gxf::Pose3D>(kNameExtrinsics);
      })
      .assign_to(message.extrinsics)
      .log_error("Failed to get '%s' from message entity", kNameExtrinsics)
      .and_then([&]() {
        return message.entity.get<int64_t>(kNameSequenceNumber);
      })
      .assign_to(message.sequence_number)
      .log_error("Failed to get '%s' from message entity", kNameSequenceNumber)
      .and_then([&]() {
        return message.entity.get<gxf::Timestamp>();
      })
      .assign_to(message.timestamp)
      .log_error("Failed to get timestamp from message entity %ld", message.entity.eid())
      .substitute(message);
}
#define CREATE_CAMERA_MESSAGE(FORMAT)                                                         \
template gxf::Expected<CameraMessageParts> CreateCameraMessage<FORMAT>(                       \
    gxf_context_t context, uint32_t width, uint32_t height, gxf::SurfaceLayout layout,        \
    gxf::MemoryStorageType storage_type, gxf::Handle<gxf::Allocator> allocator, bool padded)
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_CUSTOM);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420_ER);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420_709);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420_709_ER);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_ER);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_709);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_709_ER);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_BGRA);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_ARGB);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_ABGR);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBX);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_BGRX);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_XRGB);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_XBGR);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_R8_G8_B8);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_B8_G8_R8);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY16);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32F);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB16);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR16);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB32);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR32);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_R16_G16_B16);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_B16_G16_R16);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_R32_G32_B32);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_B32_G32_R32);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_NV24);
CREATE_CAMERA_MESSAGE(gxf::VideoFormat::GXF_VIDEO_FORMAT_NV24_ER);
}  // namespace gxf
}  // namespace nvidia
