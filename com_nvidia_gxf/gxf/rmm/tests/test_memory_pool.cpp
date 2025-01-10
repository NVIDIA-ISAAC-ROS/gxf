/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <random>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "gxf/core/gxf.h"
#include "gxf/rmm/rmm_allocator.hpp"
#include "gxf/std/gems/utils/storage_size.hpp"
#include "gxf/std/unbounded_allocator.hpp"

namespace {
#ifdef __aarch64__
const char* kBlockSize = "1MB";
#else
const char*  kBlockSize = "8MB";
#endif
}

namespace nvidia {
namespace gxf {

size_t sizeInBytes(std::string size, gxf_uid_t& cid) {
  const auto maybe = StorageSize::ParseStorageSizeString(size, cid);
  if (!maybe) {
    GXF_LOG_ERROR("Failed to parse storage string %s", size.c_str());
    return 0;
  }
  return maybe.value();
}

TEST(MemoryPool, UnboundedMemoryHost) {
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
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::UnboundedAllocator", &tid), GXF_SUCCESS);

  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test", &cid), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);

  UnboundedAllocator* mp = static_cast<UnboundedAllocator*>(pointer);
  mp->initialize();

  std::vector<uint8_t*> pointers;
  const auto blockSize = sizeInBytes(kBlockSize, cid);
  ASSERT_NE(blockSize, -1);

  for (uint64_t i = 0; i < 600; i++) {
    if (i % 2 == 0) {
      // remove
      if (pointers.empty())
        continue;
      uint8_t* result = pointers.back();
      ASSERT_EQ(mp->free_abi(result), GXF_SUCCESS);
      pointers.pop_back();
    } else {
      // add
      if (pointers.size() < kNumBlocks) {
        ASSERT_EQ(mp->is_available_abi(blockSize), GXF_SUCCESS);

        uint8_t* pointer;
        ASSERT_EQ(mp->allocate_abi(blockSize, 0 /* Host */, reinterpret_cast<void**>(&pointer)),
                  GXF_SUCCESS);
        ASSERT_NE(pointer, nullptr);
        pointers.push_back(pointer);
      }
    }
  }

  mp->deinitialize();

  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(MemoryPool, UnboundedMemoryDevice) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::UnboundedAllocator", &tid), GXF_SUCCESS);

  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test", &cid), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);

  UnboundedAllocator* mp = static_cast<UnboundedAllocator*>(pointer);
  mp->initialize();

  std::vector<uint8_t*> pointers;
  const auto blockSize = sizeInBytes(kBlockSize, cid);
  ASSERT_NE(blockSize, -1);


  for (uint64_t i = 0; i < 600; i++) {
    if (i % 2 != 0) {
      // remove
      if (pointers.empty())
        continue;
      std::uniform_int_distribution<uint64_t> idx(0, pointers.size() - 1);
      uint8_t* result = pointers.back();
      ASSERT_EQ(mp->free_abi(result), GXF_SUCCESS);
      pointers.pop_back();
    } else {
      // add
      ASSERT_EQ(mp->is_available_abi(blockSize), GXF_SUCCESS);

      uint8_t* pointer;
      ASSERT_EQ(mp->allocate_abi(blockSize, 1 /* Device */, reinterpret_cast<void**>(&pointer)),
                GXF_SUCCESS);
      ASSERT_NE(pointer, nullptr);
      pointers.push_back(pointer);
    }
  }

  mp->deinitialize();

  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(MemoryPool, RMMInitialSizeVerify) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
  constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::RMMAllocator", &tid), GXF_SUCCESS);

  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test", &cid), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);

  const char* kDevicePoolInitialSize = "32MB";  // 32 MB device initial pool size
  const char* kHostPoolInitialSize = "100MB";   // 100 MB host initial pool size
  GXF_ASSERT_SUCCESS(
      GxfParameterSetStr(context, cid, "device_memory_initial_size", kDevicePoolInitialSize));
  GXF_ASSERT_SUCCESS(
      GxfParameterSetStr(context, cid, "host_memory_initial_size", kHostPoolInitialSize));

  RMMAllocator* mp = static_cast<RMMAllocator*>(pointer);
  mp->initialize();
  const auto invalidTypeVal = mp->get_pool_size(MemoryStorageType::kSystem);
  ASSERT_EQ(ToResultCode(invalidTypeVal), GXF_ARGUMENT_INVALID);

  const auto deviceInitialSize = mp->get_pool_size(MemoryStorageType::kDevice);
  const auto hostInitialSize = mp->get_pool_size(MemoryStorageType::kHost);
  ASSERT_EQ(deviceInitialSize.value(), sizeInBytes(kDevicePoolInitialSize, cid));
  ASSERT_EQ(hostInitialSize.value(), sizeInBytes(kHostPoolInitialSize, cid));
}

TEST(MemoryPool, RMMMaxDevicePoolSize) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
  constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::RMMAllocator", &tid), GXF_SUCCESS);

  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test", &cid), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);

  const char* kDevicePoolInitialSize = "4MB";  // 4 MB device initial pool size
  const char* kDevicePoolMaxSize = "8MB";  // 8 MB device max pool size

  GXF_ASSERT_SUCCESS(
      GxfParameterSetStr(context, cid, "device_memory_initial_size", kDevicePoolInitialSize));
  GXF_ASSERT_SUCCESS(
      GxfParameterSetStr(context, cid, "device_memory_max_size", kDevicePoolMaxSize));

  RMMAllocator* mp = static_cast<RMMAllocator*>(pointer);
  mp->initialize();

  std::vector<uint8_t*> pointers;
  auto blockSize = StorageSize::toBytesFromMB(2);  // 2 MB block size

  // Make the device memory pool full
  const auto deviceMaxSize = sizeInBytes(kDevicePoolMaxSize, cid);
  ASSERT_NE(deviceMaxSize, 0);
  for (auto i = 0; i < (decltype(i))(deviceMaxSize / blockSize); ++i) {
    uint8_t* pointer;
    ASSERT_EQ(mp->allocate_abi(blockSize, 1 /* Device */, reinterpret_cast<void**>(&pointer)),
              GXF_SUCCESS);
    ASSERT_NE(pointer, nullptr);
    pointers.push_back(pointer);
  }

  // try to add more memory in memory pool
  uint8_t* pointer_invalid;
  ASSERT_EQ(mp->is_rmm_available_abi(blockSize, MemoryStorageType::kDevice), GXF_FAILURE);
  ASSERT_EQ(mp->allocate_abi(blockSize, 1, reinterpret_cast<void**>(&pointer_invalid)),
            GXF_FAILURE);
  ASSERT_NE(pointer, nullptr);

  for (const auto& ptr : pointers) {
    ASSERT_EQ(mp->free_abi(ptr), GXF_SUCCESS);
  }
}

TEST(MemoryPool, RMMMaxHostPoolSize) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
  constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::RMMAllocator", &tid), GXF_SUCCESS);

  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test", &cid), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);

  const char* kHostPoolInitialSize = "4MB";  // 4 MB device initial pool size
  const char* kHostPoolMaxSize = "8MB";   // 8 MB host max pool size

  GXF_ASSERT_SUCCESS(
      GxfParameterSetStr(context, cid, "host_memory_initial_size", kHostPoolInitialSize));
  GXF_ASSERT_SUCCESS(
      GxfParameterSetStr(context, cid, "host_memory_max_size", kHostPoolMaxSize));

  RMMAllocator* mp = static_cast<RMMAllocator*>(pointer);
  mp->initialize();

  std::vector<uint8_t*> pointers;
  size_t blockSize = StorageSize::toBytesFromMB(2);  // 2 MB block size

  // Make the device memory pool full
  const auto hostMaxSize = sizeInBytes(kHostPoolMaxSize, cid);
  ASSERT_NE(hostMaxSize, 0);
  for (auto i = 0; i < (decltype(i))(hostMaxSize / blockSize); ++i) {
    uint8_t* pointer;
    ASSERT_EQ(mp->allocate_abi(blockSize, 0 /* Host */, reinterpret_cast<void**>(&pointer)),
              GXF_SUCCESS);
    ASSERT_NE(pointer, nullptr);
    pointers.push_back(pointer);
  }

  // try to add more memory in memory pool
  uint8_t* pointer_invalid;
  ASSERT_EQ(mp->is_rmm_available_abi(blockSize, MemoryStorageType::kHost), GXF_FAILURE);
  ASSERT_EQ(mp->allocate_abi(blockSize, 0 /* Host */, reinterpret_cast<void**>(&pointer_invalid)),
            GXF_FAILURE);
  ASSERT_NE(pointer, nullptr);

  for (const auto& ptr : pointers) {
    ASSERT_EQ(mp->free_abi(ptr), GXF_SUCCESS);
  }
}

TEST(MemoryPool, RMMHostMemory) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::RMMAllocator", &tid), GXF_SUCCESS);

  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test", &cid), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);

  RMMAllocator* mp = static_cast<RMMAllocator*>(pointer);
  mp->initialize();

  std::vector<uint8_t*> pointers;
  const auto blockSize = sizeInBytes(kBlockSize, cid);
  ASSERT_NE(blockSize, 0);

  for (uint64_t i = 0; i < 600; i++) {
    if (i % 2 != 0) {
      // remove
      if (pointers.empty())
        continue;
      std::uniform_int_distribution<uint64_t> idx(0, pointers.size() - 1);
      uint8_t* result = pointers.back();
      ASSERT_EQ(mp->free_abi(result), GXF_SUCCESS);
      pointers.pop_back();
    } else {
      // add
      ASSERT_EQ(mp->is_rmm_available_abi(blockSize, MemoryStorageType::kHost), GXF_SUCCESS);

      uint8_t* pointer;
      ASSERT_EQ(mp->allocate_abi(blockSize, 0 /* Host */, reinterpret_cast<void**>(&pointer)),
                GXF_SUCCESS);
      ASSERT_NE(pointer, nullptr);
      pointers.push_back(pointer);
    }
  }

  mp->deinitialize();

  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(MemoryPool, RMMDeviceMemory) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
  constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::RMMAllocator", &tid), GXF_SUCCESS);

  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test", &cid), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);

  RMMAllocator* mp = static_cast<RMMAllocator*>(pointer);
  mp->initialize();

  std::vector<uint8_t*> pointers;
  const auto blockSize = sizeInBytes(kBlockSize, cid);
  ASSERT_NE(blockSize, 0);

  for (uint64_t i = 0; i < 600; i++) {
    if (i % 2 != 0) {
      // remove
      if (pointers.empty())
        continue;
      uint8_t* result = pointers.back();
      ASSERT_EQ(mp->free_abi(result), GXF_SUCCESS);
      pointers.pop_back();
    } else {
      // add
      ASSERT_EQ(mp->is_rmm_available_abi(blockSize, MemoryStorageType::kDevice), GXF_SUCCESS);

      uint8_t* pointer;
      ASSERT_EQ(mp->allocate_abi(blockSize, 1 /* Device */, reinterpret_cast<void**>(&pointer)),
                GXF_SUCCESS);
      ASSERT_NE(pointer, nullptr);
      pointers.push_back(pointer);
    }
  }

  mp->deinitialize();

  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(MemoryPool, RMMDeviceMemoryAsyncAPIWithSingleStream) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
  constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::RMMAllocator", &tid), GXF_SUCCESS);

  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test", &cid), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);

  RMMAllocator* mp = static_cast<RMMAllocator*>(pointer);
  mp->initialize();

  std::vector<uint8_t*> pointers;

  // Create a CUDA stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  const auto blockSize = sizeInBytes(kBlockSize, cid);
  ASSERT_NE(blockSize, 0);

  for (uint64_t i = 0; i < 600; i++) {
    if (i % 2 != 0) {
      // remove
      if (pointers.empty())
        continue;
      uint8_t* result = pointers.back();
      ASSERT_EQ(mp->free_async_abi(result, stream), GXF_SUCCESS);
      pointers.pop_back();
    } else {
      // add
      ASSERT_EQ(mp->is_rmm_available_abi(blockSize, MemoryStorageType::kDevice), GXF_SUCCESS);

      uint8_t* pointer;
      ASSERT_EQ(mp->allocate_async_abi(blockSize, reinterpret_cast<void**>(&pointer), stream),
                GXF_SUCCESS);
      ASSERT_NE(pointer, nullptr);
      pointers.push_back(pointer);
    }
  }

  // Destroy the CUDA stream
  cudaStreamDestroy(stream);
  mp->deinitialize();

  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(MemoryPool, RMMDeviceMemoryAsyncAPIWithMultipleStream) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
  constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::RMMAllocator", &tid), GXF_SUCCESS);

  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test", &cid), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);

  RMMAllocator* mp = static_cast<RMMAllocator*>(pointer);
  mp->initialize();

  std::vector<uint8_t*> pointers;

  // Create a CUDA stream
  cudaStream_t stream;
  const auto blockSize = sizeInBytes(kBlockSize, cid);
  ASSERT_NE(blockSize, 0);

  for (uint64_t i = 0; i < 600; i++) {
    if (i % 2 != 0) {
      // remove
      if (pointers.empty())
        continue;
      uint8_t* result = pointers.back();
      ASSERT_EQ(mp->free_async_abi(result, stream), GXF_SUCCESS);
      pointers.pop_back();

      // Destroy the CUDA stream
      cudaStreamDestroy(stream);
    } else {
      // add
      cudaStreamCreate(&stream);
      ASSERT_EQ(mp->is_rmm_available_abi(blockSize, MemoryStorageType::kDevice), GXF_SUCCESS);

      uint8_t* pointer;
      ASSERT_EQ(mp->allocate_async_abi(blockSize, reinterpret_cast<void**>(&pointer), stream),
                GXF_SUCCESS);
      ASSERT_NE(pointer, nullptr);
      pointers.push_back(pointer);
    }
  }

  mp->deinitialize();

  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(MemoryPool, RMMAllocateMoreThanHundredPercentDevice) {
    auto const oversize_mem_pool = rmm::percent_of_free_device_memory(105);
    const uint64_t kBlockSize = oversize_mem_pool;

    gxf_context_t context;
    ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
    constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
    const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
    ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

    gxf_uid_t eid;
    const GxfEntityCreateInfo entity_create_info = {0};
    ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

    gxf_tid_t tid;
    ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::RMMAllocator", &tid), GXF_SUCCESS);

    gxf_uid_t cid;
    ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test", &cid), GXF_SUCCESS);

    void* pointer;
    ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);

    RMMAllocator* mp = static_cast<RMMAllocator*>(pointer);
    mp->initialize();
    // add
    ASSERT_EQ(mp->is_rmm_available_abi(kBlockSize, MemoryStorageType::kDevice), GXF_FAILURE);

    uint8_t* mem_pointer;
    ASSERT_EQ(mp->allocate_abi(kBlockSize, 1 /* Device */, reinterpret_cast<void**>(&mem_pointer)),
              GXF_FAILURE);

    ASSERT_EQ(mp->free_abi(mem_pointer), GXF_FAILURE);

    mp->deinitialize();

    ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(MemoryPool, RMMTwoLargeBuffersDevice) {
  auto two_large_buffers = []() {
    gxf_context_t context;
    ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
    constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
    const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
    ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

    gxf_uid_t eid;
    const GxfEntityCreateInfo entity_create_info = {0};
    ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

    gxf_tid_t tid;
    ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::RMMAllocator", &tid), GXF_SUCCESS);

    gxf_uid_t cid;
    ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test", &cid), GXF_SUCCESS);

    void* pointer;
    ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);

    RMMAllocator* mp = static_cast<RMMAllocator*>(pointer);
    mp->initialize();

    const auto deviceInitialSize = mp->get_pool_size(MemoryStorageType::kDevice);
    const auto kBlockSize = deviceInitialSize.value() / 2;

    // add
    ASSERT_EQ(mp->is_rmm_available_abi(kBlockSize, MemoryStorageType::kDevice), GXF_SUCCESS);

    uint8_t* mem_pointer;
    ASSERT_EQ(mp->allocate_abi(kBlockSize, 1 /* Device */, reinterpret_cast<void**>(&mem_pointer)),
              GXF_SUCCESS);
    ASSERT_NE(mem_pointer, nullptr);

    ASSERT_EQ(mp->free_abi(mem_pointer), GXF_SUCCESS);

    mp->deinitialize();

    ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
  };
  EXPECT_NO_THROW(two_large_buffers());
}

TEST(MemoryPool, RMMInvalidArgAllocateAbi) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
  constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::RMMAllocator", &tid), GXF_SUCCESS);

  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test", &cid), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);

  RMMAllocator* mp = static_cast<RMMAllocator*>(pointer);
  mp->initialize();
  const auto blockSize = sizeInBytes(kBlockSize, cid);
  ASSERT_NE(blockSize, 0);

  // add
  ASSERT_EQ(mp->is_available_abi(blockSize), GXF_SUCCESS);

  uint8_t* mem_pointer = nullptr;
  // Passing incompatible device type in argument --> 2
  ASSERT_EQ(mp->allocate_abi(blockSize, 2, reinterpret_cast<void**>(&mem_pointer)),
            GXF_ARGUMENT_INVALID);
  ASSERT_EQ(mem_pointer, nullptr);

  ASSERT_EQ(mp->free_abi(mem_pointer), GXF_FAILURE);

  mp->deinitialize();

  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(MemoryPool, RMMInvalidArgFreeAsyncAbi) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
  constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::RMMAllocator", &tid), GXF_SUCCESS);

  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test", &cid), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);

  RMMAllocator* mp = static_cast<RMMAllocator*>(pointer);
  mp->initialize();

  // Create a CUDA stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  uint8_t* result = nullptr;
  ASSERT_EQ(mp->free_async_abi(result, stream), GXF_FAILURE);

  // Destroy the CUDA stream
  cudaStreamDestroy(stream);
  mp->deinitialize();

  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

}  // namespace gxf
}  // namespace nvidia
