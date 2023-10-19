/*
Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/block_memory_pool.hpp"

#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "gxf/core/gxf.h"

namespace nvidia {
namespace gxf {

TEST(MemoryPool, Test1) {
  constexpr uint64_t kBlockSize = 1024;
  constexpr uint64_t kNumBlocks = 5;

  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

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

  Allocator* mp = static_cast<Allocator*>(pointer);
  mp->initialize();

  std::mt19937 rng;
  std::uniform_int_distribution<uint32_t> coin(0, 1);

  std::vector<uint8_t*> pointers;

  for (uint64_t i = 0; i < 500; i++) {
    if (coin(rng) == 0) {
      // remove
      if (pointers.empty()) continue;
      std::uniform_int_distribution<uint64_t> idx(0, pointers.size() - 1);
      const uint64_t index = idx(rng);
      uint8_t* result = pointers[index];
      ASSERT_EQ(mp->free_abi(result), GXF_SUCCESS);
      pointers.erase(pointers.begin() + index);
    } else {
      // add
      if (pointers.size() < kNumBlocks) {
        ASSERT_EQ(mp->is_available_abi(kBlockSize), GXF_SUCCESS);

        uint8_t* pointer;
        ASSERT_EQ(mp->allocate_abi(kBlockSize, 0 /* Host */, reinterpret_cast<void**>(&pointer)),
                  GXF_SUCCESS);
        ASSERT_NE(pointer, nullptr);
        pointers.push_back(pointer);
      } else {
        ASSERT_EQ(mp->is_available_abi(kBlockSize), GXF_FAILURE);
      }
    }
  }

  mp->deinitialize();

  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

// Tests allocation and free methods of the BlockMemoryPool component for the
// kSystem memory storage type
TEST(MemoryPool, StorageTypeSystem) {
  constexpr uint64_t kBlockSize = 1024;
  constexpr uint64_t kNumBlocks = 5;

  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

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
  ASSERT_EQ(GxfParameterSetInt32(context, cid, "storage_type", 2), GXF_SUCCESS);

  Allocator* mp = static_cast<Allocator*>(pointer);
  mp->initialize();

  std::mt19937 rng;
  std::uniform_int_distribution<uint32_t> coin(0, 1);

  std::vector<uint8_t*> pointers;

  for (uint64_t i = 0; i < 500; i++) {
    if (coin(rng) == 0) {
      // choose a random pointer from the list of allocated pointers and free it
      if (pointers.empty()) continue;
      std::uniform_int_distribution<uint64_t> idx(0, pointers.size() - 1);
      const uint64_t index = idx(rng);
      uint8_t* result = pointers[index];
      ASSERT_EQ(mp->free_abi(result), GXF_SUCCESS);
      pointers.erase(pointers.begin() + index);
    } else {
      // allocate
      if (pointers.size() < kNumBlocks) {
        ASSERT_EQ(mp->is_available_abi(kBlockSize), GXF_SUCCESS);

        uint8_t* pointer;
        ASSERT_EQ(mp->allocate_abi(kBlockSize, 2 /* System */, reinterpret_cast<void**>(&pointer)),
                  GXF_SUCCESS);
        ASSERT_NE(pointer, nullptr);
        pointers.push_back(pointer);
      } else {
        ASSERT_EQ(mp->is_available_abi(kBlockSize), GXF_FAILURE);
      }
    }
  }

  mp->deinitialize();

  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

// Tests whether requested size is checked against available block size
TEST(MemoryPool, BlockSize) {
  constexpr uint64_t kBlockSize = 1024;
  constexpr uint64_t kNumBlocks = 5;

  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

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
  ASSERT_EQ(GxfParameterSetInt32(context, cid, "storage_type", 2), GXF_SUCCESS);

  Allocator* mp = static_cast<Allocator*>(pointer);
  mp->initialize();

  // actual test
  uint8_t* pointer1;
  ASSERT_EQ(mp->allocate_abi(kBlockSize, 2, reinterpret_cast<void**>(&pointer1)),
            GXF_SUCCESS);

  uint8_t* pointer2;
  ASSERT_EQ(mp->allocate_abi(kBlockSize + 1, 2, reinterpret_cast<void**>(&pointer2)),
            GXF_ARGUMENT_INVALID);

  mp->deinitialize();

  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

}  // namespace gxf
}  // namespace nvidia
