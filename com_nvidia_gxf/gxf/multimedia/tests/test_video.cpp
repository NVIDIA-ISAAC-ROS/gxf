/*
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

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

TEST(video, createFrame) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  constexpr const char* kExtensions[] = {
      "gxf/std/libgxf_std.so",
      "gxf/multimedia/libgxf_multimedia.so",
  };
  const GxfLoadExtensionsInfo info_1{kExtensions, 2, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &info_1), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::UnboundedAllocator", &tid), GXF_SUCCESS);
  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "allocator", &cid), GXF_SUCCESS);
  ASSERT_EQ(GxfParameterSetInt32(context, cid, "storage_type", 2), GXF_SUCCESS);

  auto allocator = Handle<Allocator>::Create(context, cid);
  ASSERT_EQ(allocator.has_value(), true);

  VideoBuffer* frame = new VideoBuffer();

  // For YUV420 format variants
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_YUV420>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 38400);

  // For YUV420 format variants unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_YUV420>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 15000);

  // For YUV444 multiplanar format variants
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_NV24>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);

  // For YUV444 multiplanar format variants unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_NV24>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 30000);


  // For video format with custom stride
  std::array<ColorPlane, 3> yuv{ColorPlane("Y", 1, 512), ColorPlane("U", 1, 126),
                                ColorPlane("V", 1, 126)};
  VideoFormatSize<VideoFormat::GXF_VIDEO_FORMAT_YUV420> yuv_type_trait;
  uint64_t size = yuv_type_trait.size(100, 100, yuv, false);
  std::vector<ColorPlane> yuv_filled{yuv.begin(), yuv.end()};
  VideoBufferInfo buffer_info{100, 100, VideoFormat::GXF_VIDEO_FORMAT_YUV420, yuv_filled,
                            SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR};
  result = frame->resizeCustom(buffer_info, size, MemoryStorageType::kHost, allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 63800);

  // For NV12 format variants
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_NV12>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 38400);

  // For NV12 format variants unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_NV12>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 15000);

  // For RGBA format variants
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGBA>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);

  // For RGBA format variants unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGBA>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 40000);

  // For RGB format variants
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGB>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGB16>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 76800);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGB32>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 128000);

  // For RGB format variants unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGB>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 30000);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGB16>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 60000);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGB32>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 120000);

  // For RGB multiplanar format variants
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R8_G8_B8>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 76800);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R16_G16_B16>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 76800);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R32_G32_B32>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 153600);

  // For RGB multiplanar format variants unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R8_G8_B8>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 30000);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R16_G16_B16>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 60000);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R32_G32_B32>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 120000);

  // For RGBD multiplanar format variants
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R8_G8_B8_D8>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 102400);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R16_G16_B16_D16>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 102400);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R32_G32_B32_D32>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 204800);

  // For RGBD multiplanar format variants unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R8_G8_B8_D8>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 40000);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R16_G16_B16_D16>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 80000);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R32_G32_B32_D32>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 160000);

  // For RGBD single plane format variants
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGBD8>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGBD16>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 102400);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGBD32>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 179200);

  // For RGBD single plane format variants unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGBD8>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 40000);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGBD16>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 80000);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGBD32>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 160000);

  // For Depth single plane format variants
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_D32F>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_D64F>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 102400);

  // For Depth single plane format variants unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_D32F>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 40000);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_D64F>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 80000);

  // For GRAY formats
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_GRAY>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 25600);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_GRAY16>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 25600);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_GRAY32>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_GRAY32F>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);

  // For GRAY formats unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_GRAY>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 10000);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_GRAY16>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 20000);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_GRAY32>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 40000);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_GRAY32F>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 40000);

// For Bayer RAW16 formats
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RAW16_RGGB>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 102400);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RAW16_BGGR>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 102400);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RAW16_GRBG>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 102400);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RAW16_GBRG>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 102400);

// For Bayer RAW16 formats unaligned
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RAW16_RGGB>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 80000);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RAW16_BGGR>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 80000);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RAW16_GRBG>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 80000);

  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RAW16_GBRG>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator.value(), false);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 80000);

  delete frame;
  ASSERT_EQ(allocator.value()->deinitialize(), GXF_SUCCESS);
  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(video, wrapMemory) {
  constexpr uint64_t kBlockSize = 1024000;
  constexpr uint64_t kNumBlocks = 1;

  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  constexpr const char* kExtensions[] = {
      "gxf/std/libgxf_std.so",
      "gxf/multimedia/libgxf_multimedia.so",
  };
  const GxfLoadExtensionsInfo info_1{kExtensions, 2, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &info_1), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::BlockMemoryPool", &tid), GXF_SUCCESS);

  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test", &cid), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);

  ASSERT_EQ(GxfParameterSetUInt64(context, cid, "block_size", kBlockSize), GXF_SUCCESS);
  ASSERT_EQ(GxfParameterSetUInt64(context, cid, "num_blocks", kNumBlocks), GXF_SUCCESS);

  auto maybe_allocator = Handle<Allocator>::Create(context, cid);
  ASSERT_TRUE(maybe_allocator.has_value());

  auto allocator = maybe_allocator.value();

  ASSERT_EQ(allocator->initialize(), GXF_SUCCESS);

  ASSERT_EQ(allocator->is_available_abi(kBlockSize), GXF_SUCCESS);

  void *pointer_ = this;
  bool release_func_params_match = false;
  MemoryBuffer::release_function_t release_func = [pointer_,
                                                 &release_func_params_match](void* pointer) {
    release_func_params_match = (pointer == pointer_);
    return Success;
  };

  VideoBuffer* frame = new VideoBuffer();
  static constexpr uint64_t FRAME_SIZE = 123456;

  VideoBufferInfo info;
  info.color_format = VideoFormat::GXF_VIDEO_FORMAT_NV12;
  frame->wrapMemory(info, FRAME_SIZE, MemoryStorageType::kDevice, pointer_, release_func);

  ASSERT_EQ(pointer_, frame->pointer());
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);
  ASSERT_EQ(frame->size(), FRAME_SIZE);
  ASSERT_EQ(frame->video_frame_info().color_format, VideoFormat::GXF_VIDEO_FORMAT_NV12);

  delete frame;

  ASSERT_TRUE(release_func_params_match);
  release_func_params_match = false;

  frame = new VideoBuffer();

  frame->wrapMemory(info, FRAME_SIZE, MemoryStorageType::kDevice, pointer_, release_func);

  frame->resize<VideoFormat::GXF_VIDEO_FORMAT_NV12>(
      16, 16, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost, allocator);

  ASSERT_EQ(allocator->is_available_abi(kBlockSize), GXF_FAILURE);

  ASSERT_TRUE(release_func_params_match);
  release_func_params_match = false;

  frame->wrapMemory(info, FRAME_SIZE, MemoryStorageType::kDevice, pointer_, release_func);

  ASSERT_EQ(allocator->is_available_abi(kBlockSize), GXF_SUCCESS);

  frame->wrapMemory(info, FRAME_SIZE, MemoryStorageType::kDevice, pointer_, release_func);

  ASSERT_TRUE(release_func_params_match);
  release_func_params_match = false;
  ASSERT_EQ(allocator->is_available_abi(kBlockSize), GXF_SUCCESS);

  delete frame;
  ASSERT_TRUE(release_func_params_match);

  ASSERT_EQ(allocator->deinitialize(), GXF_SUCCESS);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(video, moveToTensor) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  constexpr const char* kExtensions[] = {
      "gxf/std/libgxf_std.so",
      "gxf/multimedia/libgxf_multimedia.so",
  };
  const GxfLoadExtensionsInfo info_1{kExtensions, 2, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &info_1), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t allocator_tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::UnboundedAllocator", &allocator_tid), GXF_SUCCESS);
  gxf_uid_t allocator_cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, allocator_tid, "allocator", &allocator_cid), GXF_SUCCESS);
  ASSERT_EQ(GxfParameterSetInt32(context, allocator_cid, "storage_type", 2), GXF_SUCCESS);

  auto maybe_allocator = Handle<Allocator>::Create(context, allocator_cid);
  ASSERT_EQ(maybe_allocator.has_value(), true);

  auto allocator = maybe_allocator.value();

  // 8-bit image
  VideoBuffer* frame = new VideoBuffer();
  auto result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_RGB>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kHost,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 51200);
  byte* source_ptr = frame->pointer();

  gxf_tid_t tensor_tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::Tensor", &tensor_tid), GXF_SUCCESS);
  gxf_uid_t tensor_cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tensor_tid, "tensor", &tensor_cid), GXF_SUCCESS);

  auto maybe_tensor = Handle<Tensor>::Create(context, tensor_cid);
  ASSERT_EQ(maybe_tensor.has_value(), true);

  auto tensor = maybe_tensor.value();
  result = frame->moveToTensor(tensor);
  GXF_ASSERT_SUCCESS(ToResultCode(result));

  ASSERT_EQ(tensor->size(), 51200);
  ASSERT_EQ(tensor->rank(), 2);
  ASSERT_EQ(tensor->element_type(), PrimitiveType::kUnsigned8);
  ASSERT_EQ(tensor->bytes_per_element(), 3);
  ASSERT_EQ(tensor->storage_type(),  MemoryStorageType::kHost);

  ASSERT_EQ(nullptr, frame->pointer());
  ASSERT_EQ(source_ptr, tensor->pointer());

  // 16-bit image
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_GRAY16>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kDevice,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 25600);
  source_ptr = frame->pointer();

  result = frame->moveToTensor(tensor);
  GXF_ASSERT_SUCCESS(ToResultCode(result));

  ASSERT_EQ(tensor->size(), 25600);
  ASSERT_EQ(tensor->rank(), 2);
  ASSERT_EQ(tensor->element_type(), PrimitiveType::kUnsigned16);
  ASSERT_EQ(tensor->bytes_per_element(), 2);
  ASSERT_EQ(tensor->storage_type(),  MemoryStorageType::kDevice);

  ASSERT_EQ(nullptr, frame->pointer());
  ASSERT_EQ(source_ptr, tensor->pointer());

  // 32-bit image
  result = frame->resize<VideoFormat::GXF_VIDEO_FORMAT_R32_G32_B32>(
      100, 100, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, MemoryStorageType::kSystem,
      allocator);
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 153600);
  source_ptr = frame->pointer();

  result = frame->moveToTensor(tensor);
  GXF_ASSERT_SUCCESS(ToResultCode(result));

  ASSERT_EQ(tensor->size(), 153600);
  ASSERT_EQ(tensor->rank(), 3);
  ASSERT_EQ(tensor->element_type(), PrimitiveType::kUnsigned32);
  ASSERT_EQ(tensor->bytes_per_element(), 12);
  ASSERT_EQ(tensor->storage_type(),  MemoryStorageType::kSystem);

  ASSERT_EQ(nullptr, frame->pointer());
  ASSERT_EQ(source_ptr, tensor->pointer());

  delete frame;
  tensor->~Tensor();
  ASSERT_EQ(allocator->deinitialize(), GXF_SUCCESS);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);

}

TEST(video, createFromTensor) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  constexpr const char* kExtensions[] = {
      "gxf/std/libgxf_std.so",
      "gxf/multimedia/libgxf_multimedia.so",
  };
  const GxfLoadExtensionsInfo info_1{kExtensions, 2, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &info_1), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t allocator_tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::UnboundedAllocator", &allocator_tid), GXF_SUCCESS);
  gxf_uid_t allocator_cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, allocator_tid, "allocator", &allocator_cid), GXF_SUCCESS);
  ASSERT_EQ(GxfParameterSetInt32(context, allocator_cid, "storage_type", 2), GXF_SUCCESS);

  auto maybe_allocator = Handle<Allocator>::Create(context, allocator_cid);
  ASSERT_EQ(maybe_allocator.has_value(), true);

  auto allocator = maybe_allocator.value();

  gxf_tid_t tensor_tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::Tensor", &tensor_tid), GXF_SUCCESS);
  gxf_uid_t tensor_cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tensor_tid, "tensor", &tensor_cid), GXF_SUCCESS);

  // 1. rank=2, channels=1, VideoFormat=GXF_VIDEO_FORMAT_RGB, PrimitiveType=kUnsigned8 -> success
  auto maybe_tensor = Handle<Tensor>::Create(context, tensor_cid);
  ASSERT_EQ(maybe_tensor.has_value(), true);
  auto tensor = maybe_tensor.value();

  tensor->reshapeCustom(Shape({100, 100}), PrimitiveType::kUnsigned8, 1,
                        Unexpected{GXF_UNINITIALIZED_VALUE}, MemoryStorageType::kHost, allocator);

  ASSERT_EQ(tensor->size(), 100 * 100);

  VideoBuffer* frame = new VideoBuffer();
  auto result = frame->createFromTensor<VideoFormat::GXF_VIDEO_FORMAT_RGB>(
    tensor, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(Handle<Tensor>::Null(), tensor);
  delete frame;

  // 2. rank=3, channels=3, VideoFormat=GXF_VIDEO_FORMAT_R16_G16_B16, PrimitiveType=kUnsigned16 -> success
  maybe_tensor = Handle<Tensor>::Create(context, tensor_cid);
  ASSERT_EQ(maybe_tensor.has_value(), true);
  tensor = maybe_tensor.value();

  tensor->reshapeCustom(Shape({100, 100, 3}), PrimitiveType::kUnsigned16, 1,
                        Unexpected{GXF_UNINITIALIZED_VALUE}, MemoryStorageType::kHost, allocator);

  ASSERT_EQ(tensor->size(), 100 * 100 * 3);

  frame = new VideoBuffer();
  result = frame->createFromTensor<VideoFormat::GXF_VIDEO_FORMAT_R16_G16_B16>(
    tensor, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(Handle<Tensor>::Null(), tensor);
  delete frame;

  // 3. rank=3, channels=4, VideoFormat=GXF_VIDEO_FORMAT_R16_G16_B16, PrimitiveType=kUnsigned16 -> failure
  maybe_tensor = Handle<Tensor>::Create(context, tensor_cid);
  ASSERT_EQ(maybe_tensor.has_value(), true);
  tensor = maybe_tensor.value();

  tensor->reshapeCustom(Shape({100, 100, 4}), PrimitiveType::kUnsigned16, 1,
                        Unexpected{GXF_UNINITIALIZED_VALUE}, MemoryStorageType::kHost, allocator);

  ASSERT_EQ(tensor->size(), 100 * 100 * 4);

  frame = new VideoBuffer();
  result = frame->createFromTensor<VideoFormat::GXF_VIDEO_FORMAT_R16_G16_B16>(
    tensor, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  GXF_ASSERT_NE(ToResultCode(result), GXF_SUCCESS);
  delete frame;

  // 4. rank=4, channels=3, VideoFormat=GXF_VIDEO_FORMAT_R16_G16_B16, PrimitiveType=kUnsigned16 -> failure
  maybe_tensor = Handle<Tensor>::Create(context, tensor_cid);
  ASSERT_EQ(maybe_tensor.has_value(), true);
  tensor = maybe_tensor.value();

  tensor->reshapeCustom(Shape({100, 100, 3, 3}), PrimitiveType::kUnsigned16, 1,
                        Unexpected{GXF_UNINITIALIZED_VALUE}, MemoryStorageType::kHost, allocator);

  ASSERT_EQ(tensor->size(), 100 * 100 * 3 * 3);

  frame = new VideoBuffer();
  result = frame->createFromTensor<VideoFormat::GXF_VIDEO_FORMAT_R16_G16_B16>(
    tensor, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  GXF_ASSERT_NE(ToResultCode(result), GXF_SUCCESS);
  delete frame;

  // 5. rank=2, channels=1, VideoFormat=GXF_VIDEO_FORMAT_RGB, PrimitiveType=kUnsigned16 -> failure
  maybe_tensor = Handle<Tensor>::Create(context, tensor_cid);
  ASSERT_EQ(maybe_tensor.has_value(), true);
  tensor = maybe_tensor.value();

  tensor->reshapeCustom(Shape({100, 100}), PrimitiveType::kUnsigned16, 1,
                        Unexpected{GXF_UNINITIALIZED_VALUE}, MemoryStorageType::kHost, allocator);

  ASSERT_EQ(tensor->size(), 100 * 100);

  frame = new VideoBuffer();
  result = frame->createFromTensor<VideoFormat::GXF_VIDEO_FORMAT_RGB>(
    tensor, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  GXF_ASSERT_NE(ToResultCode(result), GXF_SUCCESS);
  delete frame;

  // 6. rank=2, channels=2, VideoFormat=GXF_VIDEO_FORMAT_NV12, PrimitiveType=kUnsigned16 -> failure
  maybe_tensor = Handle<Tensor>::Create(context, tensor_cid);
  ASSERT_EQ(maybe_tensor.has_value(), true);
  tensor = maybe_tensor.value();

  tensor->reshapeCustom(Shape({100, 100}), PrimitiveType::kUnsigned16, 1,
                        Unexpected{GXF_UNINITIALIZED_VALUE}, MemoryStorageType::kHost, allocator);

  ASSERT_EQ(tensor->size(), 100 * 100);

  frame = new VideoBuffer();
  result = frame->createFromTensor<VideoFormat::GXF_VIDEO_FORMAT_NV12>(
    tensor, SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);

  GXF_ASSERT_NE(ToResultCode(result), GXF_SUCCESS);
  delete frame;

  // clean up
  tensor->~Tensor();
  ASSERT_EQ(allocator->deinitialize(), GXF_SUCCESS);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);

}

}  // namespace gxf
}  // namespace nvidia
