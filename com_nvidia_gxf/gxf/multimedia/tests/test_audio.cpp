/*
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "common/logger.hpp"
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"
#include "gxf/multimedia/audio.hpp"

namespace nvidia {
namespace gxf {

TEST(audio, createFrame) {
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

  AudioBuffer* frame = new AudioBuffer();
  auto result = frame->resize<AudioFormat::GXF_AUDIO_FORMAT_S16LE>(
      100, 100, 10, AudioLayout::GXF_AUDIO_LAYOUT_NON_INTERLEAVED,
      MemoryStorageType::kHost, allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 20000);

  result = frame->resize<AudioFormat::GXF_AUDIO_FORMAT_F32LE>(
      100, 100, 10, AudioLayout::GXF_AUDIO_LAYOUT_NON_INTERLEAVED,
      MemoryStorageType::kHost, allocator.value());
  GXF_ASSERT_SUCCESS(ToResultCode(result));
  ASSERT_EQ(frame->size(), 40000);

  delete frame;
  ASSERT_EQ(allocator.value()->deinitialize(), GXF_SUCCESS);
  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(audio, wrapMemory) {
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

  ASSERT_EQ(allocator->is_available_abi(kBlockSize), GXF_INVALID_LIFECYCLE_STAGE);

  ASSERT_EQ(allocator->initialize(), GXF_SUCCESS);

  ASSERT_EQ(allocator->is_available_abi(kBlockSize), GXF_SUCCESS);

  void* pointer_ = this;
  bool release_func_params_match = false;
  MemoryBuffer::release_function_t release_func = [pointer_,
                                                 &release_func_params_match](void* pointer) {
    release_func_params_match = (pointer == pointer_);
    return Success;
  };

  AudioBuffer* frame = new AudioBuffer();
  static constexpr uint64_t FRAME_SIZE = 123456;

  AudioBufferInfo info;
  info.audio_layout = AudioLayout::GXF_AUDIO_LAYOUT_NON_INTERLEAVED;
  frame->wrapMemory(info, FRAME_SIZE, MemoryStorageType::kDevice, pointer_, release_func);

  ASSERT_EQ(pointer_, frame->pointer());
  ASSERT_EQ(frame->storage_type(), MemoryStorageType::kDevice);
  ASSERT_EQ(frame->size(), FRAME_SIZE);
  ASSERT_EQ(frame->audio_buffer_info().audio_layout, AudioLayout::GXF_AUDIO_LAYOUT_NON_INTERLEAVED);

  delete frame;

  ASSERT_TRUE(release_func_params_match);
  release_func_params_match = false;

  frame = new AudioBuffer();

  frame->wrapMemory(info, FRAME_SIZE, MemoryStorageType::kDevice, pointer_, release_func);

  frame->resize<AudioFormat::GXF_AUDIO_FORMAT_S16LE>(
      100, 100, 20, AudioLayout::GXF_AUDIO_LAYOUT_NON_INTERLEAVED, MemoryStorageType::kHost, allocator);

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

}  // namespace gxf
}  // namespace nvidia
