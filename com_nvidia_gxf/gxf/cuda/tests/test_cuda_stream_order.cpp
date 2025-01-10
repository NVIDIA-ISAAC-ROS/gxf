/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "gxf/core/gxf.h"
#include "gxf/cuda/stream_ordered_allocator.hpp"
#include "gxf/std/unbounded_allocator.hpp"
#include "gxf/std/gems/utils/storage_size.hpp"

namespace {
#ifdef __aarch64__
const char* kBlockSize = "1MB";
#else
const char* kBlockSize = "8MB";
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

TEST(MemoryPool, CudaHostMemory) {
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
      ASSERT_EQ(mp->is_available_abi(blockSize), GXF_SUCCESS);

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

TEST(MemoryPool, CudaDeviceMemory) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
  constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::StreamOrderedAllocator", &tid), GXF_SUCCESS);

  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test", &cid), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);

  StreamOrderedAllocator* mp = static_cast<StreamOrderedAllocator*>(pointer);
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

TEST(MemoryPool, CudaDeviceMemoryAsyncAPIWithSingleStream) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
  constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::StreamOrderedAllocator", &tid), GXF_SUCCESS);

  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test", &cid), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);

  StreamOrderedAllocator* mp = static_cast<StreamOrderedAllocator*>(pointer);
  mp->initialize();

  std::vector<uint8_t*> pointers;
  const auto blockSize = sizeInBytes(kBlockSize, cid);
  ASSERT_NE(blockSize, 0);

  // Create a CUDA stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);
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
      ASSERT_EQ(mp->is_available_abi(blockSize), GXF_SUCCESS);

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

TEST(MemoryPool, CudaDeviceMemoryAsyncAPIWithMultipleStream) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
  constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::StreamOrderedAllocator", &tid), GXF_SUCCESS);

  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test", &cid), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);

  StreamOrderedAllocator* mp = static_cast<StreamOrderedAllocator*>(pointer);
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
      ASSERT_EQ(mp->is_available_abi(blockSize), GXF_SUCCESS);

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

TEST(MemoryPool, CudaAllocateMoreThanHundredPercentDevice) {
  auto allocate_more_than_hundred_percent = []() {
    size_t free_mem = 0;
    size_t total_mem = 0;

    // Get the amount of free and total memory on the device
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
      GXF_LOG_ERROR("Error: %s", cudaGetErrorString(err));
      throw std::runtime_error(cudaGetErrorString(err));
    }

    // Calculate 105% of the free memory
    size_t oversize_mem_pool = static_cast<size_t>(total_mem * 1.05);

    const size_t blockSize = oversize_mem_pool;

    gxf_context_t context;
    ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
    constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
    const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
    ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

    gxf_uid_t eid;
    const GxfEntityCreateInfo entity_create_info = {0};
    ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

    gxf_tid_t tid;
    ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::StreamOrderedAllocator", &tid), GXF_SUCCESS);

    gxf_uid_t cid;
    ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test", &cid), GXF_SUCCESS);

    void* pointer;
    ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);

    StreamOrderedAllocator* mp = static_cast<StreamOrderedAllocator*>(pointer);
    mp->initialize();

    // add
    ASSERT_EQ(mp->is_available_abi(blockSize), GXF_FAILURE);

    uint8_t* mem_pointer;
    ASSERT_EQ(mp->allocate_abi(blockSize, 1 /* Device */, reinterpret_cast<void**>(&mem_pointer)),
              GXF_FAILURE);
    ASSERT_EQ(mem_pointer, nullptr);

    mp->deinitialize();

    ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
  };
  allocate_more_than_hundred_percent();
}

TEST(MemoryPool, CudaTwoLargeBuffersDevice) {
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
    ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::StreamOrderedAllocator", &tid), GXF_SUCCESS);

    gxf_uid_t cid;
    ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test", &cid), GXF_SUCCESS);

    void* pointer;
    ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);

    StreamOrderedAllocator* mp = static_cast<StreamOrderedAllocator*>(pointer);
    mp->initialize();

#ifdef __aarch64__
    const auto kBlockSize = "4MB";
#else
    const auto kBlockSize = "16MB";
#endif
    const auto blockSize = sizeInBytes(kBlockSize, cid);
    ASSERT_NE(blockSize, 0);
    // add
    ASSERT_EQ(mp->is_available_abi(blockSize), GXF_SUCCESS);

    uint8_t* mem_pointer;
    ASSERT_EQ(mp->allocate_abi(blockSize, 1 /* Device */, reinterpret_cast<void**>(&mem_pointer)),
              GXF_SUCCESS);
    ASSERT_NE(mem_pointer, nullptr);

    ASSERT_EQ(mp->free_abi(mem_pointer), GXF_SUCCESS);

    mp->deinitialize();

    ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
  };
  EXPECT_NO_THROW(two_large_buffers());
}

TEST(MemoryPool, InvalidGxfArgumentAllocateAbi) {
  const auto kBlockSize = "1024";

  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
  constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::StreamOrderedAllocator", &tid), GXF_SUCCESS);

  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test", &cid), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);

  StreamOrderedAllocator* mp = static_cast<StreamOrderedAllocator*>(pointer);
  mp->initialize();

  // add
  ASSERT_EQ(mp->is_available_abi(std::stoi(kBlockSize)), GXF_SUCCESS);

  uint8_t* mem_pointer = nullptr;
  // Passing incompatible device type in argument --> 2
  ASSERT_EQ(mp->allocate_abi(std::stoi(kBlockSize), 2, reinterpret_cast<void**>(&mem_pointer)),
            GXF_ARGUMENT_INVALID);
  ASSERT_EQ(mem_pointer, nullptr);

  mp->deinitialize();

  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
};

}  // namespace gxf
}  // namespace nvidia
