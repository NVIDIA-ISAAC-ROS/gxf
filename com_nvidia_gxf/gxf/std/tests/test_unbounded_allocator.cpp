/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/unbounded_allocator.hpp"

#include <cstring>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"

namespace nvidia {
namespace gxf {

namespace {

constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";

}  // namespace

TEST(TestUnboundedAllocator, CudaHost) {
  constexpr size_t kBlockSizeSmall = 100;
  constexpr size_t kBlockSizeMedium = 10000;
  constexpr size_t kBlockSizeLarge = 1000000;

  byte* buffer = new byte[kBlockSizeLarge];
  std::memset(buffer, 0xAA, kBlockSizeLarge);

  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::UnboundedAllocator", &tid), GXF_SUCCESS);
  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "allocator", &cid), GXF_SUCCESS);
  ASSERT_EQ(GxfParameterSetInt32(context, cid, "storage_type", 0), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);
  Allocator* allocator = static_cast<Allocator*>(pointer);

  ASSERT_EQ(allocator->initialize(), GXF_SUCCESS);

  ASSERT_TRUE(allocator->is_available(kBlockSizeSmall));
  auto block1 = allocator->allocate(kBlockSizeSmall, MemoryStorageType::kHost);
  ASSERT_TRUE(block1);
  std::memcpy(block1.value(), buffer, kBlockSizeSmall);
  ASSERT_EQ(std::memcmp(block1.value(), buffer, kBlockSizeSmall), 0);

  ASSERT_TRUE(allocator->is_available(kBlockSizeMedium));
  auto block2 = allocator->allocate(kBlockSizeMedium, MemoryStorageType::kHost);
  ASSERT_TRUE(block2);
  std::memcpy(block2.value(), buffer, kBlockSizeMedium);
  ASSERT_EQ(std::memcmp(block2.value(), buffer, kBlockSizeMedium), 0);

  ASSERT_TRUE(allocator->is_available(kBlockSizeLarge));
  auto block3 = allocator->allocate(kBlockSizeLarge, MemoryStorageType::kHost);
  ASSERT_TRUE(block3);
  std::memcpy(block3.value(), buffer, kBlockSizeLarge);
  ASSERT_EQ(std::memcmp(block3.value(), buffer, kBlockSizeLarge), 0);

  ASSERT_TRUE(allocator->free(block1.value()));
  ASSERT_TRUE(allocator->free(block2.value()));
  ASSERT_TRUE(allocator->free(block3.value()));

  ASSERT_EQ(allocator->deinitialize(), GXF_SUCCESS);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
  delete[] buffer;
}

TEST(TestUnboundedAllocator, CudaDevice) {
  constexpr size_t kBlockSizeSmall = 100;
  constexpr size_t kBlockSizeMedium = 10000;
  constexpr size_t kBlockSizeLarge = 1000000;

  byte* data = new byte[kBlockSizeLarge];
  byte* buffer = new byte[kBlockSizeLarge];
  std::memset(buffer, 0xAA, kBlockSizeLarge);

  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::UnboundedAllocator", &tid), GXF_SUCCESS);
  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "allocator", &cid), GXF_SUCCESS);
  ASSERT_EQ(GxfParameterSetInt32(context, cid, "storage_type", 1), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);
  Allocator* allocator = static_cast<Allocator*>(pointer);

  ASSERT_EQ(allocator->initialize(), GXF_SUCCESS);

  ASSERT_TRUE(allocator->is_available(kBlockSizeSmall));
  auto block1 = allocator->allocate(kBlockSizeSmall, MemoryStorageType::kDevice);
  ASSERT_TRUE(block1);
  std::memset(data, 0x00, kBlockSizeLarge);
  ASSERT_EQ(cudaMemcpy(block1.value(), buffer, kBlockSizeSmall, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(data, block1.value(), kBlockSizeSmall, cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(std::memcmp(data, buffer, kBlockSizeSmall), 0);

  ASSERT_TRUE(allocator->is_available(kBlockSizeMedium));
  auto block2 = allocator->allocate(kBlockSizeMedium, MemoryStorageType::kDevice);
  ASSERT_TRUE(block2);
  std::memset(data, 0x00, kBlockSizeLarge);
  ASSERT_EQ(cudaMemcpy(block2.value(), buffer, kBlockSizeMedium, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(data, block2.value(), kBlockSizeMedium, cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(std::memcmp(data, buffer, kBlockSizeMedium), 0);

  ASSERT_TRUE(allocator->is_available(kBlockSizeLarge));
  auto block3 = allocator->allocate(kBlockSizeLarge, MemoryStorageType::kDevice);
  ASSERT_TRUE(block3);
  std::memset(data, 0x00, kBlockSizeLarge);
  ASSERT_EQ(cudaMemcpy(block3.value(), buffer, kBlockSizeLarge, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(data, block3.value(), kBlockSizeLarge, cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(std::memcmp(data, buffer, kBlockSizeLarge), 0);

  ASSERT_TRUE(allocator->free(block1.value()));
  ASSERT_TRUE(allocator->free(block2.value()));
  ASSERT_TRUE(allocator->free(block3.value()));

  ASSERT_EQ(allocator->deinitialize(), GXF_SUCCESS);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
  delete[] buffer;
  delete[] data;
}

// Tests allocation and free methods of the UnboundedAllocator component for the
// kSystem memory storage type
TEST(TestUnboundedAllocator, System) {
  constexpr size_t kBlockSizeSmall = 100;
  constexpr size_t kBlockSizeMedium = 10000;
  constexpr size_t kBlockSizeLarge = 1000000;

  byte* buffer = new byte[kBlockSizeLarge];
  std::memset(buffer, 0xAA, kBlockSizeLarge);

  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::UnboundedAllocator", &tid), GXF_SUCCESS);
  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "allocator", &cid), GXF_SUCCESS);
  ASSERT_EQ(GxfParameterSetInt32(context, cid, "storage_type", 2), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);
  Allocator* allocator = static_cast<Allocator*>(pointer);

  ASSERT_EQ(allocator->initialize(), GXF_SUCCESS);

  ASSERT_TRUE(allocator->is_available(kBlockSizeSmall));
  auto block1 = allocator->allocate(kBlockSizeSmall, MemoryStorageType::kSystem);
  ASSERT_TRUE(block1);
  std::memcpy(block1.value(), buffer, kBlockSizeSmall);
  ASSERT_EQ(std::memcmp(block1.value(), buffer, kBlockSizeSmall), 0);

  ASSERT_TRUE(allocator->is_available(kBlockSizeMedium));
  auto block2 = allocator->allocate(kBlockSizeMedium, MemoryStorageType::kSystem);
  ASSERT_TRUE(block2);
  std::memcpy(block2.value(), buffer, kBlockSizeMedium);
  ASSERT_EQ(std::memcmp(block2.value(), buffer, kBlockSizeMedium), 0);

  ASSERT_TRUE(allocator->is_available(kBlockSizeLarge));
  auto block3 = allocator->allocate(kBlockSizeLarge, MemoryStorageType::kSystem);
  ASSERT_TRUE(block3);
  std::memcpy(block3.value(), buffer, kBlockSizeLarge);
  ASSERT_EQ(std::memcmp(block3.value(), buffer, kBlockSizeLarge), 0);

  ASSERT_TRUE(allocator->free(block1.value()));
  ASSERT_TRUE(allocator->free(block2.value()));
  ASSERT_TRUE(allocator->free(block3.value()));

  ASSERT_EQ(allocator->deinitialize(), GXF_SUCCESS);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
  delete[] buffer;
}

// Tests allocation and free methods of the UnboundedAllocator component for the
// kSystem memory storage type
TEST(TestUnboundedAllocator, ZeroSize) {
  constexpr size_t kBlockSizeEmpty = 0;
  constexpr size_t kBlockSizeSmall = 100;

  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::UnboundedAllocator", &tid), GXF_SUCCESS);
  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "allocator", &cid), GXF_SUCCESS);
  ASSERT_EQ(GxfParameterSetInt32(context, cid, "storage_type", 2), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);
  Allocator* allocator = static_cast<Allocator*>(pointer);


  auto block1 = allocator->allocate(kBlockSizeEmpty, MemoryStorageType::kHost);
  ASSERT_TRUE(block1);
  auto block2 = allocator->allocate(kBlockSizeSmall, MemoryStorageType::kHost);
  ASSERT_TRUE(block2);
  auto block3 = allocator->allocate(kBlockSizeEmpty, MemoryStorageType::kHost);
  ASSERT_TRUE(block3);

  ASSERT_TRUE(allocator->free(block1.value()));
  ASSERT_TRUE(allocator->free(block2.value()));
  ASSERT_TRUE(allocator->free(block3.value()));

  ASSERT_EQ(allocator->deinitialize(), GXF_SUCCESS);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

}  // namespace gxf
}  // namespace nvidia
