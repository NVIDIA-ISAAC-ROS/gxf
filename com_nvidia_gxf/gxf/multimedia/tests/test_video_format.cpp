/*
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "common/byte.hpp"
#include "common/logger.hpp"
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"
#include "gxf/multimedia/video.hpp"

namespace nvidia {
namespace gxf {

constexpr const char* kExtensions[] = {
    "gxf/std/libgxf_std.so",
    "gxf/multimedia/libgxf_multimedia.so",
};

class VideoFormatTest : public ::testing::Test {
 protected:
  void SetUp() override {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
    GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::UnboundedAllocator", &tid));
    GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "allocator", &cid));
    // GXF_ASSERT_SUCCESS(GxfParameterSetInt32(context, cid, "storage_type", 2));
    auto result = Handle<Allocator>::Create(context, cid);
    ASSERT_EQ(result.has_value(), true);
    allocator = result.value();
  }

  void TearDown() override {
    GXF_ASSERT_SUCCESS(allocator->deinitialize());
    GXF_ASSERT_SUCCESS(GxfEntityDestroy(context, eid));
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
  }

  gxf_context_t context = kNullContext;
  const GxfLoadExtensionsInfo info{kExtensions, 1, nullptr, 0, nullptr};
  gxf_uid_t eid = kNullUid;
  const GxfEntityCreateInfo entity_create_info = {0};
  gxf_tid_t tid = GxfTidNull();
  gxf_uid_t cid = kNullUid;
  Handle<Allocator> allocator;
  const uint32_t kHeight = 100;
  const uint32_t kWidth = 120;
};

//  Specifies YUV420 multi-planar variants
TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_YUV420) {

  VideoBuffer* frame = new VideoBuffer();

  // For YUV420 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_YUV420>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 38400);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_YUV420);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 3);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  25600);
  ASSERT_EQ(info.color_planes[0].color_space, "Y");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[1].stride, 128);
  ASSERT_EQ(info.color_planes[1].offset, 25600);
  ASSERT_EQ(info.color_planes[1].width, kWidth/2);
  ASSERT_EQ(info.color_planes[1].height, kHeight/2);
  ASSERT_EQ(info.color_planes[1].size, 6400);
  ASSERT_EQ(info.color_planes[1].color_space,"U");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[2].stride, 128);
  ASSERT_EQ(info.color_planes[2].offset, 32000);
  ASSERT_EQ(info.color_planes[2].width, kWidth/2);
  ASSERT_EQ(info.color_planes[2].height, kHeight/2);
  ASSERT_EQ(info.color_planes[2].size, 6400);
  ASSERT_EQ(info.color_planes[2].color_space,"V");

  // For YUV420 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_YUV420>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 18000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_YUV420);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 3);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kWidth);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  12000);
  ASSERT_EQ(info.color_planes[0].color_space, "Y");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[1].stride, kWidth/2);
  ASSERT_EQ(info.color_planes[1].offset, 12000);
  ASSERT_EQ(info.color_planes[1].width, kWidth/2);
  ASSERT_EQ(info.color_planes[1].height, kHeight/2);
  ASSERT_EQ(info.color_planes[1].size, 3000);
  ASSERT_EQ(info.color_planes[1].color_space,"U");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[2].stride, kWidth/2);
  ASSERT_EQ(info.color_planes[2].offset, 15000);
  ASSERT_EQ(info.color_planes[2].width, kWidth/2);
  ASSERT_EQ(info.color_planes[2].height, kHeight/2);
  ASSERT_EQ(info.color_planes[2].size, 3000);
  ASSERT_EQ(info.color_planes[2].color_space,"V");

  delete frame;

}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_YUV420_ER) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_YUV420_ER format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_YUV420_ER>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 38400);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_YUV420_ER);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 3);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  25600);
  ASSERT_EQ(info.color_planes[0].color_space, "Y");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[1].stride, 128);
  ASSERT_EQ(info.color_planes[1].offset, 25600);
  ASSERT_EQ(info.color_planes[1].width, kWidth/2);
  ASSERT_EQ(info.color_planes[1].height, kHeight/2);
  ASSERT_EQ(info.color_planes[1].size, 6400);
  ASSERT_EQ(info.color_planes[1].color_space,"U");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[2].stride, 128);
  ASSERT_EQ(info.color_planes[2].offset, 32000);
  ASSERT_EQ(info.color_planes[2].width, kWidth/2);
  ASSERT_EQ(info.color_planes[2].height, kHeight/2);
  ASSERT_EQ(info.color_planes[2].size, 6400);
  ASSERT_EQ(info.color_planes[2].color_space,"V");

  // For GXF_VIDEO_FORMAT_YUV420_ER format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_YUV420_ER>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 18000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_YUV420_ER);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 3);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kWidth);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  12000);
  ASSERT_EQ(info.color_planes[0].color_space, "Y");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[1].stride, kWidth/2);
  ASSERT_EQ(info.color_planes[1].offset, 12000);
  ASSERT_EQ(info.color_planes[1].width, kWidth/2);
  ASSERT_EQ(info.color_planes[1].height, kHeight/2);
  ASSERT_EQ(info.color_planes[1].size, 3000);
  ASSERT_EQ(info.color_planes[1].color_space,"U");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[2].stride, kWidth/2);
  ASSERT_EQ(info.color_planes[2].offset, 15000);
  ASSERT_EQ(info.color_planes[2].width, kWidth/2);
  ASSERT_EQ(info.color_planes[2].height, kHeight/2);
  ASSERT_EQ(info.color_planes[2].size, 3000);
  ASSERT_EQ(info.color_planes[2].color_space,"V");

  delete frame;

}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_YUV420_709) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_YUV420_709 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_YUV420_709>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 38400);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_YUV420_709);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 3);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  25600);
  ASSERT_EQ(info.color_planes[0].color_space, "Y");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[1].stride, 128);
  ASSERT_EQ(info.color_planes[1].offset, 25600);
  ASSERT_EQ(info.color_planes[1].width, kWidth/2);
  ASSERT_EQ(info.color_planes[1].height, kHeight/2);
  ASSERT_EQ(info.color_planes[1].size, 6400);
  ASSERT_EQ(info.color_planes[1].color_space,"U");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[2].stride, 128);
  ASSERT_EQ(info.color_planes[2].offset, 32000);
  ASSERT_EQ(info.color_planes[2].width, kWidth/2);
  ASSERT_EQ(info.color_planes[2].height, kHeight/2);
  ASSERT_EQ(info.color_planes[2].size, 6400);
  ASSERT_EQ(info.color_planes[2].color_space,"V");

  // For GXF_VIDEO_FORMAT_YUV420_709 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_YUV420_709>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 18000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_YUV420_709);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 3);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kWidth);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  12000);
  ASSERT_EQ(info.color_planes[0].color_space, "Y");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[1].stride, kWidth/2);
  ASSERT_EQ(info.color_planes[1].offset, 12000);
  ASSERT_EQ(info.color_planes[1].width, kWidth/2);
  ASSERT_EQ(info.color_planes[1].height, kHeight/2);
  ASSERT_EQ(info.color_planes[1].size, 3000);
  ASSERT_EQ(info.color_planes[1].color_space,"U");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[2].stride, kWidth/2);
  ASSERT_EQ(info.color_planes[2].offset, 15000);
  ASSERT_EQ(info.color_planes[2].width, kWidth/2);
  ASSERT_EQ(info.color_planes[2].height, kHeight/2);
  ASSERT_EQ(info.color_planes[2].size, 3000);
  ASSERT_EQ(info.color_planes[2].color_space,"V");

  delete frame;

}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_YUV420_709_ER) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_YUV420_709_ER format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_YUV420_709_ER>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 38400);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_YUV420_709_ER);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 3);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  25600);
  ASSERT_EQ(info.color_planes[0].color_space, "Y");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[1].stride, 128);
  ASSERT_EQ(info.color_planes[1].offset, 25600);
  ASSERT_EQ(info.color_planes[1].width, kWidth/2);
  ASSERT_EQ(info.color_planes[1].height, kHeight/2);
  ASSERT_EQ(info.color_planes[1].size, 6400);
  ASSERT_EQ(info.color_planes[1].color_space,"U");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[2].stride, 128);
  ASSERT_EQ(info.color_planes[2].offset, 32000);
  ASSERT_EQ(info.color_planes[2].width, kWidth/2);
  ASSERT_EQ(info.color_planes[2].height, kHeight/2);
  ASSERT_EQ(info.color_planes[2].size, 6400);
  ASSERT_EQ(info.color_planes[2].color_space,"V");

  // For GXF_VIDEO_FORMAT_YUV420_709_ER format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_YUV420_709_ER>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 18000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_YUV420_709_ER);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 3);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kWidth);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  12000);
  ASSERT_EQ(info.color_planes[0].color_space, "Y");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[1].stride, kWidth/2);
  ASSERT_EQ(info.color_planes[1].offset, 12000);
  ASSERT_EQ(info.color_planes[1].width, kWidth/2);
  ASSERT_EQ(info.color_planes[1].height, kHeight/2);
  ASSERT_EQ(info.color_planes[1].size, 3000);
  ASSERT_EQ(info.color_planes[1].color_space,"U");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[2].stride, kWidth/2);
  ASSERT_EQ(info.color_planes[2].offset, 15000);
  ASSERT_EQ(info.color_planes[2].width, kWidth/2);
  ASSERT_EQ(info.color_planes[2].height, kHeight/2);
  ASSERT_EQ(info.color_planes[2].size, 3000);
  ASSERT_EQ(info.color_planes[2].color_space,"V");

  delete frame;

}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_NV12) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_NV12 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_NV12>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 38400);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_NV12);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 2);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  25600);
  ASSERT_EQ(info.color_planes[0].color_space, "Y");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[1].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[1].offset, 25600);
  ASSERT_EQ(info.color_planes[1].width, kWidth/2);
  ASSERT_EQ(info.color_planes[1].height, kHeight/2);
  ASSERT_EQ(info.color_planes[1].size, 12800);
  ASSERT_EQ(info.color_planes[1].color_space,"UV");

  // For GXF_VIDEO_FORMAT_NV12 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_NV12>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 18000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_NV12);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 2);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kWidth);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  12000);
  ASSERT_EQ(info.color_planes[0].color_space, "Y");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[1].stride, kWidth);
  ASSERT_EQ(info.color_planes[1].offset, 12000);
  ASSERT_EQ(info.color_planes[1].width, kWidth/2);
  ASSERT_EQ(info.color_planes[1].height, kHeight/2);
  ASSERT_EQ(info.color_planes[1].size, 6000);
  ASSERT_EQ(info.color_planes[1].color_space,"UV");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_NV12_ER) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_NV12_ER format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_NV12_ER>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 38400);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_NV12_ER);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 2);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  25600);
  ASSERT_EQ(info.color_planes[0].color_space, "Y");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[1].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[1].offset, 25600);
  ASSERT_EQ(info.color_planes[1].width, kWidth/2);
  ASSERT_EQ(info.color_planes[1].height, kHeight/2);
  ASSERT_EQ(info.color_planes[1].size, 12800);
  ASSERT_EQ(info.color_planes[1].color_space,"UV");

  // For GXF_VIDEO_FORMAT_NV12_ER format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_NV12_ER>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 18000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_NV12_ER);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 2);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kWidth);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  12000);
  ASSERT_EQ(info.color_planes[0].color_space, "Y");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[1].stride, kWidth);
  ASSERT_EQ(info.color_planes[1].offset, 12000);
  ASSERT_EQ(info.color_planes[1].width, kWidth/2);
  ASSERT_EQ(info.color_planes[1].height, kHeight/2);
  ASSERT_EQ(info.color_planes[1].size, 6000);
  ASSERT_EQ(info.color_planes[1].color_space,"UV");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_NV12_709) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_NV12_709 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_NV12_709>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 38400);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_NV12_709);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 2);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  25600);
  ASSERT_EQ(info.color_planes[0].color_space, "Y");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[1].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[1].offset, 25600);
  ASSERT_EQ(info.color_planes[1].width, kWidth/2);
  ASSERT_EQ(info.color_planes[1].height, kHeight/2);
  ASSERT_EQ(info.color_planes[1].size, 12800);
  ASSERT_EQ(info.color_planes[1].color_space,"UV");

  // For GXF_VIDEO_FORMAT_NV12_709 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_NV12_709>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 18000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_NV12_709);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 2);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kWidth);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  12000);
  ASSERT_EQ(info.color_planes[0].color_space, "Y");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[1].stride, kWidth);
  ASSERT_EQ(info.color_planes[1].offset, 12000);
  ASSERT_EQ(info.color_planes[1].width, kWidth/2);
  ASSERT_EQ(info.color_planes[1].height, kHeight/2);
  ASSERT_EQ(info.color_planes[1].size, 6000);
  ASSERT_EQ(info.color_planes[1].color_space,"UV");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_NV12_709_ER) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_NV12_709_ER format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_NV12_709_ER>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 38400);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_NV12_709_ER);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 2);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  25600);
  ASSERT_EQ(info.color_planes[0].color_space, "Y");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[1].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[1].offset, 25600);
  ASSERT_EQ(info.color_planes[1].width, kWidth/2);
  ASSERT_EQ(info.color_planes[1].height, kHeight/2);
  ASSERT_EQ(info.color_planes[1].size, 12800);
  ASSERT_EQ(info.color_planes[1].color_space,"UV");

  // For GXF_VIDEO_FORMAT_NV12_709_ER format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_NV12_709_ER>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 18000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_NV12_709_ER);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 2);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kWidth);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  12000);
  ASSERT_EQ(info.color_planes[0].color_space, "Y");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[1].stride, kWidth);
  ASSERT_EQ(info.color_planes[1].offset, 12000);
  ASSERT_EQ(info.color_planes[1].width, kWidth/2);
  ASSERT_EQ(info.color_planes[1].height, kHeight/2);
  ASSERT_EQ(info.color_planes[1].size, 6000);
  ASSERT_EQ(info.color_planes[1].color_space,"UV");

  delete frame;
}

// Specifies YUV444 multi-planar variants

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_NV24) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_NV24 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_NV24>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_NV24);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 2);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  25600);
  ASSERT_EQ(info.color_planes[0].color_space, "Y");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[1].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[1].offset, 25600);
  ASSERT_EQ(info.color_planes[1].width, kWidth);
  ASSERT_EQ(info.color_planes[1].height, kHeight);
  ASSERT_EQ(info.color_planes[1].size, 25600);
  ASSERT_EQ(info.color_planes[1].color_space,"UV");

  // For GXF_VIDEO_FORMAT_NV24 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_NV24>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 36000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_NV24);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 2);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kWidth);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  12000);
  ASSERT_EQ(info.color_planes[0].color_space, "Y");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[1].stride, 240);
  ASSERT_EQ(info.color_planes[1].offset, 12000);
  ASSERT_EQ(info.color_planes[1].width, kWidth);
  ASSERT_EQ(info.color_planes[1].height, kHeight);
  ASSERT_EQ(info.color_planes[1].size, 24000);
  ASSERT_EQ(info.color_planes[1].color_space,"UV");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_NV24_ER) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_NV24_ER format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_NV24_ER>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_NV24_ER);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 2);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  25600);
  ASSERT_EQ(info.color_planes[0].color_space, "Y");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[1].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[1].offset, 25600);
  ASSERT_EQ(info.color_planes[1].width, kWidth);
  ASSERT_EQ(info.color_planes[1].height, kHeight);
  ASSERT_EQ(info.color_planes[1].size, 25600);
  ASSERT_EQ(info.color_planes[1].color_space,"UV");

  // For GXF_VIDEO_FORMAT_NV24_ER format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_NV24_ER>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 36000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_NV24_ER);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 2);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kWidth);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  12000);
  ASSERT_EQ(info.color_planes[0].color_space, "Y");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[1].stride, 240);
  ASSERT_EQ(info.color_planes[1].offset, 12000);
  ASSERT_EQ(info.color_planes[1].width, kWidth);
  ASSERT_EQ(info.color_planes[1].height, kHeight);
  ASSERT_EQ(info.color_planes[1].size, 24000);
  ASSERT_EQ(info.color_planes[1].color_space,"UV");

  delete frame;
}

// Specifies 8-8-8-8 single plane RGBX/XRGB variants

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_RGBA) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_RGBA format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGBA>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RGBA);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 512);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  51200);
  ASSERT_EQ(info.color_planes[0].color_space, "RGBA");

  // For GXF_VIDEO_FORMAT_RGBA format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGBA>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 48000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RGBA);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 480);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  48000);
  ASSERT_EQ(info.color_planes[0].color_space, "RGBA");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_BGRA) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_BGRA format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_BGRA>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_BGRA);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 512);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  51200);
  ASSERT_EQ(info.color_planes[0].color_space, "BGRA");

  // For GXF_VIDEO_FORMAT_BGRA format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_BGRA>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 48000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_BGRA);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 480);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  48000);
  ASSERT_EQ(info.color_planes[0].color_space, "BGRA");

  delete frame;
}


TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_ARGB) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_ARGB format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_ARGB>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_ARGB);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 512);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  51200);
  ASSERT_EQ(info.color_planes[0].color_space, "ARGB");

  // For GXF_VIDEO_FORMAT_ARGB format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_ARGB>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 48000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_ARGB);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 480);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  48000);
  ASSERT_EQ(info.color_planes[0].color_space, "ARGB");

  delete frame;
}


TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_ABGR) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_ABGR format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_ABGR>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_ABGR);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 512);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  51200);
  ASSERT_EQ(info.color_planes[0].color_space, "ABGR");

  // For GXF_VIDEO_FORMAT_ABGR format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_ABGR>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 48000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_ABGR);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 480);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  48000);
  ASSERT_EQ(info.color_planes[0].color_space, "ABGR");

  delete frame;
}


TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_RGBX) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_RGBX format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGBX>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RGBX);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 512);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  51200);
  ASSERT_EQ(info.color_planes[0].color_space, "RGBX");

  // For GXF_VIDEO_FORMAT_RGBX format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGBX>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 48000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RGBX);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 480);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  48000);
  ASSERT_EQ(info.color_planes[0].color_space, "RGBX");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_BGRX) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_BGRX format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_BGRX>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_BGRX);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 512);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  51200);
  ASSERT_EQ(info.color_planes[0].color_space, "BGRX");

  // For GXF_VIDEO_FORMAT_BGRX format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_BGRX>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 48000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_BGRX);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 480);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  48000);
  ASSERT_EQ(info.color_planes[0].color_space, "BGRX");

  delete frame;
}


TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_XRGB) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_XRGB format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_XRGB>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_XRGB);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 512);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  51200);
  ASSERT_EQ(info.color_planes[0].color_space, "XRGB");

  // For GXF_VIDEO_FORMAT_XRGB format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_XRGB>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 48000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_XRGB);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 480);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  48000);
  ASSERT_EQ(info.color_planes[0].color_space, "XRGB");

  delete frame;
}


TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_XBGR) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_XBGR format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_XBGR>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_XBGR);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 512);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  51200);
  ASSERT_EQ(info.color_planes[0].color_space, "XBGR");

  // For GXF_VIDEO_FORMAT_XBGR format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_XBGR>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 48000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_XBGR);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 480);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  48000);
  ASSERT_EQ(info.color_planes[0].color_space, "XBGR");

  delete frame;
}

// Specifies x-x-x(8/16/32) bit single plane RGB/BGR variants
TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_RGB) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_RGB format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGB>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RGB);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 3);
  ASSERT_EQ(info.color_planes[0].stride, 512);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  51200);
  ASSERT_EQ(info.color_planes[0].color_space, "RGB");

  // For GXF_VIDEO_FORMAT_RGB format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGB>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 36000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RGB);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 3);
  ASSERT_EQ(info.color_planes[0].stride, 360);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  36000);
  ASSERT_EQ(info.color_planes[0].color_space, "RGB");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_BGR) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_BGR format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_BGR>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_BGR);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 3);
  ASSERT_EQ(info.color_planes[0].stride, 512);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  51200);
  ASSERT_EQ(info.color_planes[0].color_space, "BGR");

  // For GXF_VIDEO_FORMAT_BGR format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_BGR>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 36000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_BGR);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 3);
  ASSERT_EQ(info.color_planes[0].stride, 360);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  36000);
  ASSERT_EQ(info.color_planes[0].color_space, "BGR");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_RGB16) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_RGB16 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGB16>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 76800);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RGB16);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 3 * 2);
  ASSERT_EQ(info.color_planes[0].stride, 768);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  76800);
  ASSERT_EQ(info.color_planes[0].color_space, "RGB");

  // For GXF_VIDEO_FORMAT_RGB16 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGB16>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 72000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RGB16);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 3 * 2);
  ASSERT_EQ(info.color_planes[0].stride, 720);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  72000);
  ASSERT_EQ(info.color_planes[0].color_space, "RGB");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_BGR16) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_BGR16 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_BGR16>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 76800);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_BGR16);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 3 * 2);
  ASSERT_EQ(info.color_planes[0].stride, 768);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  76800);
  ASSERT_EQ(info.color_planes[0].color_space, "BGR");

  // For GXF_VIDEO_FORMAT_BGR16 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_BGR16>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 72000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_BGR16);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 3 * 2);
  ASSERT_EQ(info.color_planes[0].stride, 720);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  72000);
  ASSERT_EQ(info.color_planes[0].color_space, "BGR");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_RGB32) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_RGB32 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGB32>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 153600);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RGB32);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 3 * 4);
  ASSERT_EQ(info.color_planes[0].stride, 1536);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  153600);
  ASSERT_EQ(info.color_planes[0].color_space, "RGB");

  // For GXF_VIDEO_FORMAT_RGB32 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGB32>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 144000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RGB32);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 3 * 4);
  ASSERT_EQ(info.color_planes[0].stride, 1440);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  144000);
  ASSERT_EQ(info.color_planes[0].color_space, "RGB");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_BGR32) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_BGR32 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_BGR32>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 153600);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_BGR32);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 3 * 4);
  ASSERT_EQ(info.color_planes[0].stride, 1536);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  153600);
  ASSERT_EQ(info.color_planes[0].color_space, "BGR");

  // For GXF_VIDEO_FORMAT_BGR32 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_BGR32>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 144000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_BGR32);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 3 * 4);
  ASSERT_EQ(info.color_planes[0].stride, 1440);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  144000);
  ASSERT_EQ(info.color_planes[0].color_space, "BGR");

  delete frame;
}

// Specifies x-x-x(8/16/32) bit multi planar RGB/BGR variants
TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_R8_G8_B8) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_R8_G8_B8 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R8_G8_B8>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 76800);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_R8_G8_B8);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 3);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  25600);
  ASSERT_EQ(info.color_planes[0].color_space, "R");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[1].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[1].offset, 25600);
  ASSERT_EQ(info.color_planes[1].width, kWidth);
  ASSERT_EQ(info.color_planes[1].height, kHeight);
  ASSERT_EQ(info.color_planes[1].size, 25600);
  ASSERT_EQ(info.color_planes[1].color_space,"G");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[2].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[2].offset, 51200);
  ASSERT_EQ(info.color_planes[2].width, kWidth);
  ASSERT_EQ(info.color_planes[2].height, kHeight);
  ASSERT_EQ(info.color_planes[2].size, 25600);
  ASSERT_EQ(info.color_planes[2].color_space,"B");

  // For GXF_VIDEO_FORMAT_R8_G8_B8 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R8_G8_B8>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 36000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_R8_G8_B8);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 3);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kWidth);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  12000);
  ASSERT_EQ(info.color_planes[0].color_space, "R");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[1].stride, kWidth);
  ASSERT_EQ(info.color_planes[1].offset, 12000);
  ASSERT_EQ(info.color_planes[1].width, kWidth);
  ASSERT_EQ(info.color_planes[1].height, kHeight);
  ASSERT_EQ(info.color_planes[1].size, 12000);
  ASSERT_EQ(info.color_planes[1].color_space,"G");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[2].stride, kWidth);
  ASSERT_EQ(info.color_planes[2].offset, 24000);
  ASSERT_EQ(info.color_planes[2].width, kWidth);
  ASSERT_EQ(info.color_planes[2].height, kHeight);
  ASSERT_EQ(info.color_planes[2].size, 12000);
  ASSERT_EQ(info.color_planes[2].color_space,"B");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_B8_G8_R8) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_B8_G8_R8 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_B8_G8_R8>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 76800);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_B8_G8_R8);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 3);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  25600);
  ASSERT_EQ(info.color_planes[0].color_space, "B");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[1].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[1].offset, 25600);
  ASSERT_EQ(info.color_planes[1].width, kWidth);
  ASSERT_EQ(info.color_planes[1].height, kHeight);
  ASSERT_EQ(info.color_planes[1].size, 25600);
  ASSERT_EQ(info.color_planes[1].color_space,"G");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[2].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[2].offset, 51200);
  ASSERT_EQ(info.color_planes[2].width, kWidth);
  ASSERT_EQ(info.color_planes[2].height, kHeight);
  ASSERT_EQ(info.color_planes[2].size, 25600);
  ASSERT_EQ(info.color_planes[2].color_space,"R");

  // For GXF_VIDEO_FORMAT_B8_G8_R8 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_B8_G8_R8>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 36000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_B8_G8_R8);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 3);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kWidth);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  12000);
  ASSERT_EQ(info.color_planes[0].color_space, "B");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[1].stride, kWidth);
  ASSERT_EQ(info.color_planes[1].offset, 12000);
  ASSERT_EQ(info.color_planes[1].width, kWidth);
  ASSERT_EQ(info.color_planes[1].height, kHeight);
  ASSERT_EQ(info.color_planes[1].size, 12000);
  ASSERT_EQ(info.color_planes[1].color_space,"G");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[2].stride, kWidth);
  ASSERT_EQ(info.color_planes[2].offset, 24000);
  ASSERT_EQ(info.color_planes[2].width, kWidth);
  ASSERT_EQ(info.color_planes[2].height, kHeight);
  ASSERT_EQ(info.color_planes[2].size, 12000);
  ASSERT_EQ(info.color_planes[2].color_space,"R");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_R16_G16_B16) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_R16_G16_B16 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R16_G16_B16>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 76800);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_R16_G16_B16);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 3);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[0].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  25600);
  ASSERT_EQ(info.color_planes[0].color_space, "R");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[1].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[1].offset, 25600);
  ASSERT_EQ(info.color_planes[1].width, kWidth);
  ASSERT_EQ(info.color_planes[1].height, kHeight);
  ASSERT_EQ(info.color_planes[1].size, 25600);
  ASSERT_EQ(info.color_planes[1].color_space,"G");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[2].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[2].offset, 51200);
  ASSERT_EQ(info.color_planes[2].width, kWidth);
  ASSERT_EQ(info.color_planes[2].height, kHeight);
  ASSERT_EQ(info.color_planes[2].size, 25600);
  ASSERT_EQ(info.color_planes[2].color_space,"B");

  // For GXF_VIDEO_FORMAT_R16_G16_B16 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R16_G16_B16>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 72000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_R16_G16_B16);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 3);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[0].stride, 240);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  24000);
  ASSERT_EQ(info.color_planes[0].color_space, "R");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[1].stride, 240);
  ASSERT_EQ(info.color_planes[1].offset, 24000);
  ASSERT_EQ(info.color_planes[1].width, kWidth);
  ASSERT_EQ(info.color_planes[1].height, kHeight);
  ASSERT_EQ(info.color_planes[1].size, 24000);
  ASSERT_EQ(info.color_planes[1].color_space,"G");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[2].stride, 240);
  ASSERT_EQ(info.color_planes[2].offset, 48000);
  ASSERT_EQ(info.color_planes[2].width, kWidth);
  ASSERT_EQ(info.color_planes[2].height, kHeight);
  ASSERT_EQ(info.color_planes[2].size, 24000);
  ASSERT_EQ(info.color_planes[2].color_space,"B");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_B16_G16_R16) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_B16_G16_R16 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_B16_G16_R16>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 76800);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_B16_G16_R16);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 3);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[0].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  25600);
  ASSERT_EQ(info.color_planes[0].color_space, "B");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[1].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[1].offset, 25600);
  ASSERT_EQ(info.color_planes[1].width, kWidth);
  ASSERT_EQ(info.color_planes[1].height, kHeight);
  ASSERT_EQ(info.color_planes[1].size, 25600);
  ASSERT_EQ(info.color_planes[1].color_space,"G");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[2].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[2].offset, 51200);
  ASSERT_EQ(info.color_planes[2].width, kWidth);
  ASSERT_EQ(info.color_planes[2].height, kHeight);
  ASSERT_EQ(info.color_planes[2].size, 25600);
  ASSERT_EQ(info.color_planes[2].color_space,"R");

  // For GXF_VIDEO_FORMAT_B16_G16_R16 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_B16_G16_R16>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 72000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_B16_G16_R16);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 3);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[0].stride, 240);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  24000);
  ASSERT_EQ(info.color_planes[0].color_space, "B");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[1].stride, 240);
  ASSERT_EQ(info.color_planes[1].offset, 24000);
  ASSERT_EQ(info.color_planes[1].width, kWidth);
  ASSERT_EQ(info.color_planes[1].height, kHeight);
  ASSERT_EQ(info.color_planes[1].size, 24000);
  ASSERT_EQ(info.color_planes[1].color_space,"G");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[2].stride, 240);
  ASSERT_EQ(info.color_planes[2].offset, 48000);
  ASSERT_EQ(info.color_planes[2].width, kWidth);
  ASSERT_EQ(info.color_planes[2].height, kHeight);
  ASSERT_EQ(info.color_planes[2].size, 24000);
  ASSERT_EQ(info.color_planes[2].color_space,"R");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_R32_G32_B32) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_R32_G32_B32 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R32_G32_B32>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 153600);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_R32_G32_B32);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 3);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 512);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  51200);
  ASSERT_EQ(info.color_planes[0].color_space, "R");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[1].stride, 512);
  ASSERT_EQ(info.color_planes[1].offset, 51200);
  ASSERT_EQ(info.color_planes[1].width, kWidth);
  ASSERT_EQ(info.color_planes[1].height, kHeight);
  ASSERT_EQ(info.color_planes[1].size, 51200);
  ASSERT_EQ(info.color_planes[1].color_space,"G");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[2].stride, 512);
  ASSERT_EQ(info.color_planes[2].offset, 102400);
  ASSERT_EQ(info.color_planes[2].width, kWidth);
  ASSERT_EQ(info.color_planes[2].height, kHeight);
  ASSERT_EQ(info.color_planes[2].size, 51200);
  ASSERT_EQ(info.color_planes[2].color_space,"B");

  // For GXF_VIDEO_FORMAT_R32_G32_B32 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R32_G32_B32>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 144000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_R32_G32_B32);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 3);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 480);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  48000);
  ASSERT_EQ(info.color_planes[0].color_space, "R");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[1].stride, 480);
  ASSERT_EQ(info.color_planes[1].offset, 48000);
  ASSERT_EQ(info.color_planes[1].width, kWidth);
  ASSERT_EQ(info.color_planes[1].height, kHeight);
  ASSERT_EQ(info.color_planes[1].size, 48000);
  ASSERT_EQ(info.color_planes[1].color_space,"G");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[2].stride, 480);
  ASSERT_EQ(info.color_planes[2].offset, 96000);
  ASSERT_EQ(info.color_planes[2].width, kWidth);
  ASSERT_EQ(info.color_planes[2].height, kHeight);
  ASSERT_EQ(info.color_planes[2].size, 48000);
  ASSERT_EQ(info.color_planes[2].color_space,"B");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_B32_G32_R32) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_B32_G32_R32 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_B32_G32_R32>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 153600);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_B32_G32_R32);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 3);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 512);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  51200);
  ASSERT_EQ(info.color_planes[0].color_space, "B");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[1].stride, 512);
  ASSERT_EQ(info.color_planes[1].offset, 51200);
  ASSERT_EQ(info.color_planes[1].width, kWidth);
  ASSERT_EQ(info.color_planes[1].height, kHeight);
  ASSERT_EQ(info.color_planes[1].size, 51200);
  ASSERT_EQ(info.color_planes[1].color_space,"G");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[2].stride, 512);
  ASSERT_EQ(info.color_planes[2].offset, 102400);
  ASSERT_EQ(info.color_planes[2].width, kWidth);
  ASSERT_EQ(info.color_planes[2].height, kHeight);
  ASSERT_EQ(info.color_planes[2].size, 51200);
  ASSERT_EQ(info.color_planes[2].color_space,"R");

  // For GXF_VIDEO_FORMAT_B32_G32_R32 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_B32_G32_R32>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 144000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_B32_G32_R32);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 3);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 480);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  48000);
  ASSERT_EQ(info.color_planes[0].color_space, "B");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[1].stride, 480);
  ASSERT_EQ(info.color_planes[1].offset, 48000);
  ASSERT_EQ(info.color_planes[1].width, kWidth);
  ASSERT_EQ(info.color_planes[1].height, kHeight);
  ASSERT_EQ(info.color_planes[1].size, 48000);
  ASSERT_EQ(info.color_planes[1].color_space,"G");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[2].stride, 480);
  ASSERT_EQ(info.color_planes[2].offset, 96000);
  ASSERT_EQ(info.color_planes[2].width, kWidth);
  ASSERT_EQ(info.color_planes[2].height, kHeight);
  ASSERT_EQ(info.color_planes[2].size, 48000);
  ASSERT_EQ(info.color_planes[2].color_space,"R");

  delete frame;
}

// Specifies x-x-x(8/16/32) bit multi planar RGBD variants
TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_R8_G8_B8_D8) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_R8_G8_B8_D8 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R8_G8_B8_D8>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 102400);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_R8_G8_B8_D8);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 4);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  25600);
  ASSERT_EQ(info.color_planes[0].color_space, "R");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[1].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[1].offset, 25600);
  ASSERT_EQ(info.color_planes[1].width, kWidth);
  ASSERT_EQ(info.color_planes[1].height, kHeight);
  ASSERT_EQ(info.color_planes[1].size, 25600);
  ASSERT_EQ(info.color_planes[1].color_space,"G");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[2].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[2].offset, 51200);
  ASSERT_EQ(info.color_planes[2].width, kWidth);
  ASSERT_EQ(info.color_planes[2].height, kHeight);
  ASSERT_EQ(info.color_planes[2].size, 25600);
  ASSERT_EQ(info.color_planes[2].color_space,"B");

  ASSERT_EQ(info.color_planes[3].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[3].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[3].offset, 76800);
  ASSERT_EQ(info.color_planes[3].width, kWidth);
  ASSERT_EQ(info.color_planes[3].height, kHeight);
  ASSERT_EQ(info.color_planes[3].size, 25600);
  ASSERT_EQ(info.color_planes[3].color_space,"D");

  // For GXF_VIDEO_FORMAT_R8_G8_B8_D8 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R8_G8_B8_D8>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 48000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_R8_G8_B8_D8);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 4);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kWidth);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  12000);
  ASSERT_EQ(info.color_planes[0].color_space, "R");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[1].stride, kWidth);
  ASSERT_EQ(info.color_planes[1].offset, 12000);
  ASSERT_EQ(info.color_planes[1].width, kWidth);
  ASSERT_EQ(info.color_planes[1].height, kHeight);
  ASSERT_EQ(info.color_planes[1].size, 12000);
  ASSERT_EQ(info.color_planes[1].color_space,"G");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[2].stride, kWidth);
  ASSERT_EQ(info.color_planes[2].offset, 24000);
  ASSERT_EQ(info.color_planes[2].width, kWidth);
  ASSERT_EQ(info.color_planes[2].height, kHeight);
  ASSERT_EQ(info.color_planes[2].size, 12000);
  ASSERT_EQ(info.color_planes[2].color_space,"B");

  ASSERT_EQ(info.color_planes[3].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[3].stride, kWidth);
  ASSERT_EQ(info.color_planes[3].offset, 36000);
  ASSERT_EQ(info.color_planes[3].width, kWidth);
  ASSERT_EQ(info.color_planes[3].height, kHeight);
  ASSERT_EQ(info.color_planes[3].size, 12000);
  ASSERT_EQ(info.color_planes[3].color_space,"D");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_R16_G16_B16_D16) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_R16_G16_B16_D16 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R16_G16_B16_D16>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 102400);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_R16_G16_B16_D16);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 4);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[0].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  25600);
  ASSERT_EQ(info.color_planes[0].color_space, "R");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[1].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[1].offset, 25600);
  ASSERT_EQ(info.color_planes[1].width, kWidth);
  ASSERT_EQ(info.color_planes[1].height, kHeight);
  ASSERT_EQ(info.color_planes[1].size, 25600);
  ASSERT_EQ(info.color_planes[1].color_space,"G");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[2].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[2].offset, 51200);
  ASSERT_EQ(info.color_planes[2].width, kWidth);
  ASSERT_EQ(info.color_planes[2].height, kHeight);
  ASSERT_EQ(info.color_planes[2].size, 25600);
  ASSERT_EQ(info.color_planes[2].color_space,"B");

  ASSERT_EQ(info.color_planes[3].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[3].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[3].offset, 76800);
  ASSERT_EQ(info.color_planes[3].width, kWidth);
  ASSERT_EQ(info.color_planes[3].height, kHeight);
  ASSERT_EQ(info.color_planes[3].size, 25600);
  ASSERT_EQ(info.color_planes[3].color_space,"D");

  // For GXF_VIDEO_FORMAT_R16_G16_B16_D16 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R16_G16_B16_D16>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 96000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_R16_G16_B16_D16);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 4);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[0].stride, 240);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  24000);
  ASSERT_EQ(info.color_planes[0].color_space, "R");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[1].stride, 240);
  ASSERT_EQ(info.color_planes[1].offset, 24000);
  ASSERT_EQ(info.color_planes[1].width, kWidth);
  ASSERT_EQ(info.color_planes[1].height, kHeight);
  ASSERT_EQ(info.color_planes[1].size, 24000);
  ASSERT_EQ(info.color_planes[1].color_space,"G");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[2].stride, 240);
  ASSERT_EQ(info.color_planes[2].offset, 48000);
  ASSERT_EQ(info.color_planes[2].width, kWidth);
  ASSERT_EQ(info.color_planes[2].height, kHeight);
  ASSERT_EQ(info.color_planes[2].size, 24000);
  ASSERT_EQ(info.color_planes[2].color_space,"B");

  ASSERT_EQ(info.color_planes[3].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[3].stride, 240);
  ASSERT_EQ(info.color_planes[3].offset, 72000);
  ASSERT_EQ(info.color_planes[3].width, kWidth);
  ASSERT_EQ(info.color_planes[3].height, kHeight);
  ASSERT_EQ(info.color_planes[3].size, 24000);
  ASSERT_EQ(info.color_planes[3].color_space,"D");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_R32_G32_B32_D32) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_R32_G32_B32_D32 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R32_G32_B32_D32>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 204800);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_R32_G32_B32_D32);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 4);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 512);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  51200);
  ASSERT_EQ(info.color_planes[0].color_space, "R");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[1].stride, 512);
  ASSERT_EQ(info.color_planes[1].offset, 51200);
  ASSERT_EQ(info.color_planes[1].width, kWidth);
  ASSERT_EQ(info.color_planes[1].height, kHeight);
  ASSERT_EQ(info.color_planes[1].size, 51200);
  ASSERT_EQ(info.color_planes[1].color_space,"G");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[2].stride, 512);
  ASSERT_EQ(info.color_planes[2].offset, 102400);
  ASSERT_EQ(info.color_planes[2].width, kWidth);
  ASSERT_EQ(info.color_planes[2].height, kHeight);
  ASSERT_EQ(info.color_planes[2].size, 51200);
  ASSERT_EQ(info.color_planes[2].color_space,"B");

  ASSERT_EQ(info.color_planes[3].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[3].stride, 512);
  ASSERT_EQ(info.color_planes[3].offset, 153600);
  ASSERT_EQ(info.color_planes[3].width, kWidth);
  ASSERT_EQ(info.color_planes[3].height, kHeight);
  ASSERT_EQ(info.color_planes[3].size, 51200);
  ASSERT_EQ(info.color_planes[3].color_space,"D");

  // For GXF_VIDEO_FORMAT_R32_G32_B32_D32 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R32_G32_B32_D32>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 192000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_R32_G32_B32_D32);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 4);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 480);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  48000);
  ASSERT_EQ(info.color_planes[0].color_space, "R");

  ASSERT_EQ(info.color_planes[1].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[1].stride, 480);
  ASSERT_EQ(info.color_planes[1].offset, 48000);
  ASSERT_EQ(info.color_planes[1].width, kWidth);
  ASSERT_EQ(info.color_planes[1].height, kHeight);
  ASSERT_EQ(info.color_planes[1].size, 48000);
  ASSERT_EQ(info.color_planes[1].color_space,"G");

  ASSERT_EQ(info.color_planes[2].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[2].stride, 480);
  ASSERT_EQ(info.color_planes[2].offset, 96000);
  ASSERT_EQ(info.color_planes[2].width, kWidth);
  ASSERT_EQ(info.color_planes[2].height, kHeight);
  ASSERT_EQ(info.color_planes[2].size, 48000);
  ASSERT_EQ(info.color_planes[2].color_space,"B");

  ASSERT_EQ(info.color_planes[3].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[3].stride, 480);
  ASSERT_EQ(info.color_planes[3].offset, 144000);
  ASSERT_EQ(info.color_planes[3].width, kWidth);
  ASSERT_EQ(info.color_planes[3].height, kHeight);
  ASSERT_EQ(info.color_planes[3].size, 48000);
  ASSERT_EQ(info.color_planes[3].color_space,"D");

  delete frame;
}

// Specifies x-x-x-x(8/16/32) bit float single plane RGBD variants
TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_RGBD8) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_RGBD8 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGBD8>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RGBD8);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1 * 4);
  ASSERT_EQ(info.color_planes[0].stride, 512);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  51200);
  ASSERT_EQ(info.color_planes[0].color_space, "RGBD");

  // For GXF_VIDEO_FORMAT_RGBD8 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGBD8>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 48000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RGBD8);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 480);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  48000);
  ASSERT_EQ(info.color_planes[0].color_space, "RGBD");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_RGBD16) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_RGBD16 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGBD16>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 102400);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RGBD16);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 2 * 4);
  ASSERT_EQ(info.color_planes[0].stride, 1024);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  102400);
  ASSERT_EQ(info.color_planes[0].color_space, "RGBD");

  // For GXF_VIDEO_FORMAT_RGBD16 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGBD16>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 96000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RGBD16);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 2 * 4);
  ASSERT_EQ(info.color_planes[0].stride, 960);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  96000);
  ASSERT_EQ(info.color_planes[0].color_space, "RGBD");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_RGBD32) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_RGBD32 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGBD32>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 204800);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RGBD32);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4 * 4);
  ASSERT_EQ(info.color_planes[0].stride, 2048);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  204800);
  ASSERT_EQ(info.color_planes[0].color_space, "RGBD");

  // For GXF_VIDEO_FORMAT_RGBD32 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGBD32>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 192000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RGBD32);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4 * 4);
  ASSERT_EQ(info.color_planes[0].stride, 1920);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  192000);
  ASSERT_EQ(info.color_planes[0].color_space, "RGBD");

  delete frame;
}

// Specifies 32/64 bit float single plane Depth
TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_D32F) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_D32F format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_D32F>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_D32F);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 512);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  51200);
  ASSERT_EQ(info.color_planes[0].color_space, "D");

  // For GXF_VIDEO_FORMAT_D32F format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_D32F>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 48000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_D32F);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 480);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  48000);
  ASSERT_EQ(info.color_planes[0].color_space, "D");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_D64F) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_D64F format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_D64F>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 102400);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_D64F);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 8);
  ASSERT_EQ(info.color_planes[0].stride, 1024);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  102400);
  ASSERT_EQ(info.color_planes[0].color_space, "D");

  // For GXF_VIDEO_FORMAT_D64F format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_D64F>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 96000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_D64F);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 8);
  ASSERT_EQ(info.color_planes[0].stride, 960);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  96000);
  ASSERT_EQ(info.color_planes[0].color_space, "D");

  delete frame;
}

// Specifies x-x-x(8/16/32) bit single plane GRAY scale
TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_GRAY) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_GRAY format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_GRAY>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 25600);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_GRAY);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  25600);
  ASSERT_EQ(info.color_planes[0].color_space, "gray");

  // For GXF_VIDEO_FORMAT_GRAY format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_GRAY>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 12000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_GRAY);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 1);
  ASSERT_EQ(info.color_planes[0].stride, 120);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  12000);
  ASSERT_EQ(info.color_planes[0].color_space, "gray");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_GRAY16) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_GRAY16 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_GRAY16>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 25600);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_GRAY16);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[0].stride, kGxfAlignValue);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  25600);
  ASSERT_EQ(info.color_planes[0].color_space, "gray");

  // For GXF_VIDEO_FORMAT_GRAY16 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_GRAY16>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 24000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_GRAY16);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 2);
  ASSERT_EQ(info.color_planes[0].stride, 240);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  24000);
  ASSERT_EQ(info.color_planes[0].color_space, "gray");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_GRAY32) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_GRAY32 format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_GRAY32>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_GRAY32);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 512);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  51200);
  ASSERT_EQ(info.color_planes[0].color_space, "gray");

  // For GXF_VIDEO_FORMAT_GRAY32 format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_GRAY32>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 48000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_GRAY32);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 480);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  48000);
  ASSERT_EQ(info.color_planes[0].color_space, "gray");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_GRAY32F) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_GRAY32F format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_GRAY32F>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_GRAY32F);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 512);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  51200);
  ASSERT_EQ(info.color_planes[0].color_space, "gray");

  // For GXF_VIDEO_FORMAT_GRAY32F format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_GRAY32F>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 48000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_GRAY32F);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4);
  ASSERT_EQ(info.color_planes[0].stride, 480);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  48000);
  ASSERT_EQ(info.color_planes[0].color_space, "gray");

  delete frame;
}

// Specifies x-x-x-x(16) bit single plane Bayer RAW16
TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_RAW16_RGGB) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_RAW16_RGGB format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RAW16_RGGB>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 102400);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RAW16_RGGB);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4 * 2);
  ASSERT_EQ(info.color_planes[0].stride, 1024);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  102400);
  ASSERT_EQ(info.color_planes[0].color_space, "Raw_RGGB");

  // For GXF_VIDEO_FORMAT_RAW16_RGGB format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RAW16_RGGB>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 96000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RAW16_RGGB);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4 * 2);
  ASSERT_EQ(info.color_planes[0].stride, 960);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  96000);
  ASSERT_EQ(info.color_planes[0].color_space, "Raw_RGGB");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_RAW16_BGGR) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_RAW16_BGGR format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RAW16_BGGR>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 102400);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RAW16_BGGR);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4 * 2);
  ASSERT_EQ(info.color_planes[0].stride, 1024);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  102400);
  ASSERT_EQ(info.color_planes[0].color_space, "Raw_BGGR");

  // For GXF_VIDEO_FORMAT_RAW16_BGGR format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RAW16_BGGR>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 96000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RAW16_BGGR);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4 * 2);
  ASSERT_EQ(info.color_planes[0].stride, 960);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  96000);
  ASSERT_EQ(info.color_planes[0].color_space, "Raw_BGGR");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_RAW16_GRBG) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_RAW16_GRBG format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RAW16_GRBG>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 102400);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RAW16_GRBG);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4 * 2);
  ASSERT_EQ(info.color_planes[0].stride, 1024);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  102400);
  ASSERT_EQ(info.color_planes[0].color_space, "Raw_GRBG");

  // For GXF_VIDEO_FORMAT_RAW16_GRBG format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RAW16_GRBG>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 96000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RAW16_GRBG);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4 * 2);
  ASSERT_EQ(info.color_planes[0].stride, 960);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  96000);
  ASSERT_EQ(info.color_planes[0].color_space, "Raw_GRBG");

  delete frame;
}

TEST_F(VideoFormatTest, GXF_VIDEO_FORMAT_RAW16_GBRG) {

  VideoBuffer* frame = new VideoBuffer();

  // For GXF_VIDEO_FORMAT_RAW16_GBRG format aligned
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RAW16_GBRG>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 102400);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kHost);

  VideoBufferInfo info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RAW16_GBRG);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4 * 2);
  ASSERT_EQ(info.color_planes[0].stride, 1024);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  102400);
  ASSERT_EQ(info.color_planes[0].color_space, "Raw_GBRG");

  // For GXF_VIDEO_FORMAT_RAW16_GBRG format unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RAW16_GBRG>(
      kWidth, kHeight, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR, MemoryStorageType::kDevice,
      allocator, false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 96000);
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);

  info = frame->video_frame_info();
  ASSERT_EQ(info.height, kHeight);
  ASSERT_EQ(info.width, kWidth);
  ASSERT_EQ(info.color_format, VideoFormat::GXF_VIDEO_FORMAT_RAW16_GBRG);
  ASSERT_EQ(info.surface_layout, SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR);

  ASSERT_EQ(info.color_planes.size(), 1);
  ASSERT_EQ(info.color_planes[0].bytes_per_pixel, 4 * 2);
  ASSERT_EQ(info.color_planes[0].stride, 960);
  ASSERT_EQ(info.color_planes[0].offset, 0);
  ASSERT_EQ(info.color_planes[0].width, kWidth);
  ASSERT_EQ(info.color_planes[0].height, kHeight);
  ASSERT_EQ(info.color_planes[0].size,  96000);
  ASSERT_EQ(info.color_planes[0].color_space, "Raw_GBRG");

  delete frame;
}

}  // namespace gxf
}  // namespace nvidia
