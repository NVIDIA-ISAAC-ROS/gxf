/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/tensor.hpp"

#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "gxf/core/gxf.h"

namespace nvidia {
namespace gxf {

TEST(Tensor, reshapeCustomAllocator) {
  constexpr uint64_t kBlockSize = 1024;
  constexpr uint64_t kNumBlocks = 1;

  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  constexpr const char* kExtensions[] = {
      "gxf/std/libgxf_std.so",
  };
  const GxfLoadExtensionsInfo info{kExtensions, 1, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &info), GXF_SUCCESS);

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

  Tensor* tensor = new Tensor();

  tensor->reshapeCustom(Shape({1, kBlockSize}), PrimitiveType::kUnsigned8, 1,
                        Unexpected{GXF_UNINITIALIZED_VALUE}, MemoryStorageType::kHost, allocator);

  ASSERT_EQ(allocator->is_available_abi(kBlockSize), GXF_FAILURE);

  delete tensor;

  tensor = new Tensor();

  ASSERT_EQ(allocator->is_available_abi(kBlockSize), GXF_SUCCESS);

  tensor->reshapeCustom(Shape({1, kBlockSize}), PrimitiveType::kUnsigned8, 1,
                        Unexpected{GXF_UNINITIALIZED_VALUE}, MemoryStorageType::kHost, allocator);

  ASSERT_EQ(allocator->is_available_abi(kBlockSize), GXF_FAILURE);

  tensor->wrapMemory(Shape({1, kBlockSize}), PrimitiveType::kUnsigned8, 1,
                     Unexpected{GXF_UNINITIALIZED_VALUE}, MemoryStorageType::kHost, nullptr,
                     nullptr);

  ASSERT_EQ(allocator->is_available_abi(kBlockSize), GXF_SUCCESS);

  tensor->reshapeCustom(Shape({1, kBlockSize}), PrimitiveType::kUnsigned8, 1,
                        Unexpected{GXF_UNINITIALIZED_VALUE}, MemoryStorageType::kHost, allocator);

  ASSERT_EQ(allocator->is_available_abi(kBlockSize), GXF_FAILURE);

  delete tensor;

  ASSERT_EQ(allocator->is_available_abi(kBlockSize), GXF_SUCCESS);

  ASSERT_EQ(allocator->deinitialize(), GXF_SUCCESS);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(Tensor, wrapMemory) {
  constexpr uint64_t kBlockSize = 1024;
  constexpr uint64_t kNumBlocks = 1;

  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  constexpr const char* kExtensions[] = {
      "gxf/std/libgxf_std.so",
  };
  const GxfLoadExtensionsInfo info{kExtensions, 1, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &info), GXF_SUCCESS);

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

  void* pointer_ = this;
  bool release_func_params_match = false;
  MemoryBuffer::release_function_t release_func = [pointer_, &release_func_params_match](void* pointer) {
    release_func_params_match = (pointer == pointer_);
    return Success;
  };

  Tensor* tensor = new Tensor();

  tensor->wrapMemory(Shape({4, kBlockSize}), PrimitiveType::kUnsigned8, 4,
                     Unexpected{GXF_UNINITIALIZED_VALUE}, MemoryStorageType::kDevice, pointer_,
                     release_func);

  ASSERT_EQ(pointer_, tensor->pointer());
  ASSERT_EQ(tensor->shape().rank(), 2);
  ASSERT_EQ(tensor->shape().dimension(0), 4);
  ASSERT_EQ(tensor->shape().dimension(1), kBlockSize);
  ASSERT_EQ(tensor->storage_type(), MemoryStorageType::kDevice);
  ASSERT_EQ(tensor->bytes_per_element(), 4);
  ASSERT_EQ(tensor->element_count(), 4 * kBlockSize);
  ASSERT_EQ(tensor->bytes_size(), 4 * kBlockSize * 4);
  ASSERT_EQ(tensor->rank(), 2);
  ASSERT_EQ(tensor->size(), 4 * kBlockSize * 4);
  ASSERT_EQ(tensor->element_type(), PrimitiveType::kUnsigned8);
  ASSERT_EQ(tensor->stride(1), 4);
  ASSERT_EQ(tensor->stride(0), 4 * kBlockSize);
  ASSERT_EQ(pointer_, tensor->data<uint8_t>().value());

  delete tensor;

  ASSERT_TRUE(release_func_params_match);
  release_func_params_match = false;

  tensor = new Tensor();


  tensor->wrapMemory(Shape({1, kBlockSize}), PrimitiveType::kUnsigned8, 1,
                     Unexpected{GXF_UNINITIALIZED_VALUE}, MemoryStorageType::kHost, pointer_,
                     release_func);

  tensor->reshapeCustom(Shape({1, kBlockSize}), PrimitiveType::kUnsigned8, 1,
                        Unexpected{GXF_UNINITIALIZED_VALUE}, MemoryStorageType::kHost, allocator);

  ASSERT_EQ(allocator->is_available_abi(kBlockSize), GXF_FAILURE);

  ASSERT_TRUE(release_func_params_match);
  release_func_params_match = false;

  tensor->wrapMemory(Shape({1, kBlockSize}), PrimitiveType::kUnsigned8, 1,
                     Unexpected{GXF_UNINITIALIZED_VALUE}, MemoryStorageType::kHost, pointer_,
                     release_func);

  ASSERT_EQ(allocator->is_available_abi(kBlockSize), GXF_SUCCESS);

  tensor->wrapMemory(Shape({1, kBlockSize}), PrimitiveType::kUnsigned8, 1,
                     Unexpected{GXF_UNINITIALIZED_VALUE}, MemoryStorageType::kHost, pointer_,
                     release_func);

  ASSERT_TRUE(release_func_params_match);
  release_func_params_match = false;
  ASSERT_EQ(allocator->is_available_abi(kBlockSize), GXF_SUCCESS);

  delete tensor;
  ASSERT_TRUE(release_func_params_match);

  ASSERT_EQ(allocator->deinitialize(), GXF_SUCCESS);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(Tensor, permute) {
  constexpr uint64_t width = 3;
  constexpr uint64_t height = 2;
  constexpr uint64_t channels = 2;
  constexpr uint64_t kBlockSize = channels * width * height;
  constexpr uint64_t kNumBlocks = 1;

  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  constexpr const char* kExtensions[] = {
      "gxf/std/libgxf_std.so",
  };
  const GxfLoadExtensionsInfo info{kExtensions, 1, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &info), GXF_SUCCESS);

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

  // Initialize array data used for testing
  std::array<uint8_t, kBlockSize> arr;
  std::iota(std::begin(arr), std::end(arr), 0);

  // Get pointer to array data
  void* pointer_ = arr.data();
  bool release_func_params_match = false;
  MemoryBuffer::release_function_t release_func = [pointer_, &release_func_params_match](void* pointer) {
    release_func_params_match = (pointer == pointer_);
    return Success;
  };

  // Initialize tensor
  Tensor* tensor = new Tensor();
  auto shape = Shape({width, height, channels});
  tensor->wrapMemory(shape, PrimitiveType::kUnsigned8, 4,
                     Unexpected{GXF_UNINITIALIZED_VALUE}, MemoryStorageType::kHost, pointer_,
                     release_func);
  // Assert
  ASSERT_EQ(pointer_, tensor->pointer());
  ASSERT_EQ(tensor->shape().rank(), 3);
  ASSERT_EQ(tensor->shape().dimension(0), width);
  ASSERT_EQ(tensor->shape().dimension(1), height);
  ASSERT_EQ(tensor->shape().dimension(2), channels);
  ASSERT_EQ(tensor->storage_type(), MemoryStorageType::kHost);
  ASSERT_EQ(tensor->bytes_per_element(), 4);
  ASSERT_EQ(tensor->element_count(), kBlockSize);
  ASSERT_EQ(tensor->bytes_size(), 4 * kBlockSize);
  ASSERT_EQ(tensor->rank(), 3);
  ASSERT_EQ(tensor->size(), 4 * kBlockSize);
  ASSERT_EQ(tensor->element_type(), PrimitiveType::kUnsigned8);
  ASSERT_EQ(tensor->stride(0), 4 * height * channels);
  ASSERT_EQ(tensor->stride(1), 4 * height);
  ASSERT_EQ(tensor->stride(2), 4);
  ASSERT_EQ(pointer_, tensor->data<uint8_t>().value());
  // Check correct element access
  std::array<uint8_t, kBlockSize> ref = {0, 1, 2, 3, 4, 5, 6, 7, 8 ,9, 10, 11};
  int32_t ir = 0, ii = 0, jj = 0, kk = 0;
  uint8_t x = 0;
  for(int32_t i = 0; i < tensor->shape().dimension(0); i++) {
    ii = i*static_cast<int32_t>(tensor->stride(0))/4;
    for(int32_t j = 0; j < tensor->shape().dimension(1); j++) {
      jj = j*static_cast<int32_t>(tensor->stride(1))/4;
      for(int32_t k = 0; k < tensor->shape().dimension(2); k++) {
        kk = k*static_cast<int32_t>(tensor->stride(2))/4;
        x = *(tensor->data<uint8_t>().value() + ii + jj + kk);
        ASSERT_EQ(ref[ir++], x);
      }
    }
  }

  // Permute (WHC) to (CHW)
  tensor->permute({2, 1, 0});
  // Assert
  ASSERT_EQ(pointer_, tensor->pointer());
  ASSERT_EQ(tensor->shape().dimension(0), channels);
  ASSERT_EQ(tensor->shape().dimension(1), height);
  ASSERT_EQ(tensor->shape().dimension(2), width);
  ASSERT_EQ(tensor->storage_type(), MemoryStorageType::kHost);
  ASSERT_EQ(tensor->bytes_per_element(), 4);
  ASSERT_EQ(tensor->element_count(), kBlockSize);
  ASSERT_EQ(tensor->bytes_size(), 4 * channels);
  ASSERT_EQ(tensor->rank(), 3);
  ASSERT_EQ(tensor->size(), 4 * kBlockSize);
  ASSERT_EQ(tensor->element_type(), PrimitiveType::kUnsigned8);
  ASSERT_EQ(tensor->stride(0), 4);
  ASSERT_EQ(tensor->stride(1), 4 * height);
  ASSERT_EQ(tensor->stride(2), 4 * height * channels);
  ASSERT_EQ(pointer_, tensor->data<uint8_t>().value());
  // Check correct element access
  ref = {0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11};
  ir = 0;
  for(int32_t i = 0; i < tensor->shape().dimension(0); i++) {
    ii = i*static_cast<int32_t>(tensor->stride(0))/4;
    for(int32_t j = 0; j < tensor->shape().dimension(1); j++) {
      jj = j*static_cast<int32_t>(tensor->stride(1))/4;
      for(int32_t k = 0; k < tensor->shape().dimension(2); k++) {
        kk = k*static_cast<int32_t>(tensor->stride(2))/4;
        x = *(tensor->data<uint8_t>().value() + ii + jj + kk);
        ASSERT_EQ(ref[ir++], x);
      }
    }
  }

  // Clean up
  delete tensor;

  ASSERT_TRUE(release_func_params_match);
  release_func_params_match = false;

  ASSERT_EQ(allocator->deinitialize(), GXF_SUCCESS);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(Tensor, noCopyReshape) {
  constexpr uint64_t width = 3;
  constexpr uint64_t height = 2;
  constexpr uint64_t channels = 2;
  constexpr uint64_t kBlockSize = channels * width * height;
  constexpr uint64_t kNumBlocks = 1;

  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  constexpr const char* kExtensions[] = {
      "gxf/std/libgxf_std.so",
  };
  const GxfLoadExtensionsInfo info{kExtensions, 1, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &info), GXF_SUCCESS);

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

  // Initialize array data used for testing
  std::array<uint8_t, kBlockSize> arr;
  std::iota(std::begin(arr), std::end(arr), 0);

  // Get pointer to array data
  void* pointer_ = arr.data();
  bool release_func_params_match = false;
  MemoryBuffer::release_function_t release_func = [pointer_, &release_func_params_match](void* pointer) {
    release_func_params_match = (pointer == pointer_);
    return Success;
  };

  // Initialize tensor
  Tensor* tensor = new Tensor();
  auto shape = Shape({width, height, channels});
  tensor->wrapMemory(shape, PrimitiveType::kUnsigned8, 4,
                     Unexpected{GXF_UNINITIALIZED_VALUE}, MemoryStorageType::kHost, pointer_,
                     release_func);
  // Assert
  ASSERT_EQ(pointer_, tensor->pointer());
  ASSERT_EQ(tensor->shape().rank(), 3);
  ASSERT_EQ(tensor->shape().dimension(0), width);
  ASSERT_EQ(tensor->shape().dimension(1), height);
  ASSERT_EQ(tensor->shape().dimension(2), channels);
  ASSERT_EQ(tensor->storage_type(), MemoryStorageType::kHost);
  ASSERT_EQ(tensor->bytes_per_element(), 4);
  ASSERT_EQ(tensor->element_count(), kBlockSize);
  ASSERT_EQ(tensor->bytes_size(), 4 * kBlockSize);
  ASSERT_EQ(tensor->rank(), 3);
  ASSERT_EQ(tensor->size(), 4 * kBlockSize);
  ASSERT_EQ(tensor->element_type(), PrimitiveType::kUnsigned8);
  ASSERT_EQ(tensor->stride(0), 4 * height * channels);
  ASSERT_EQ(tensor->stride(1), 4 * height);
  ASSERT_EQ(tensor->stride(2), 4);
  ASSERT_EQ(pointer_, tensor->data<uint8_t>().value());
  // Check correct element access
  std::array<uint8_t, kBlockSize> ref = {0, 1, 2, 3, 4, 5, 6, 7, 8 ,9, 10, 11};
  int32_t ir = 0, ii = 0, jj = 0, kk = 0;
  uint8_t x = 0;
  for(int32_t i = 0; i < tensor->shape().dimension(0); i++) {
    ii = i*static_cast<int32_t>(tensor->stride(0))/4;
    for(int32_t j = 0; j < tensor->shape().dimension(1); j++) {
      jj = j*static_cast<int32_t>(tensor->stride(1))/4;
      for(int32_t k = 0; k < tensor->shape().dimension(2); k++) {
        kk = k*static_cast<int32_t>(tensor->stride(2))/4;
        x = *(tensor->data<uint8_t>().value() + ii + jj + kk);
        ASSERT_EQ(ref[ir++], x);
      }
    }
  }

  // Reshape (W,H,C) to (C,H,W)
  tensor->noCopyReshape({channels, height, width});
  // Assert
  ASSERT_EQ(pointer_, tensor->pointer());
  ASSERT_EQ(tensor->shape().dimension(0), channels);
  ASSERT_EQ(tensor->shape().dimension(1), height);
  ASSERT_EQ(tensor->shape().dimension(2), width);
  ASSERT_EQ(tensor->storage_type(), MemoryStorageType::kHost);
  ASSERT_EQ(tensor->bytes_per_element(), 4);
  ASSERT_EQ(tensor->element_count(), kBlockSize);
  ASSERT_EQ(tensor->bytes_size(), 4 * kBlockSize);
  ASSERT_EQ(tensor->rank(), 3);
  ASSERT_EQ(tensor->size(), 4 * kBlockSize);
  ASSERT_EQ(tensor->element_type(), PrimitiveType::kUnsigned8);
  ASSERT_EQ(tensor->stride(0), 24);
  ASSERT_EQ(tensor->stride(1), 12);
  ASSERT_EQ(tensor->stride(2), 4);
  ASSERT_EQ(pointer_, tensor->data<uint8_t>().value());
  // Check correct element access
  ir = 0;
  for(int32_t i = 0; i < tensor->shape().dimension(0); i++) {
    ii = i*static_cast<int32_t>(tensor->stride(0))/4;
    for(int32_t j = 0; j < tensor->shape().dimension(1); j++) {
      jj = j*static_cast<int32_t>(tensor->stride(1))/4;
      for(int32_t k = 0; k < tensor->shape().dimension(2); k++) {
        kk = k*static_cast<int32_t>(tensor->stride(2))/4;
        x = *(tensor->data<uint8_t>().value() + ii + jj + kk);
        ASSERT_EQ(ref[ir++], x);
      }
    }
  }

  // Reshape (C,H,W) to (C,HxW)
  tensor->noCopyReshape({channels, height*width});
  // Assert
  ASSERT_EQ(pointer_, tensor->pointer());
  ASSERT_EQ(tensor->shape().dimension(0), channels);
  ASSERT_EQ(tensor->shape().dimension(1), height*width);
  ASSERT_EQ(tensor->storage_type(), MemoryStorageType::kHost);
  ASSERT_EQ(tensor->bytes_per_element(), 4);
  ASSERT_EQ(tensor->element_count(), kBlockSize);
  ASSERT_EQ(tensor->bytes_size(), 4 * kBlockSize);
  ASSERT_EQ(tensor->rank(), 2);
  ASSERT_EQ(tensor->size(), 4 * kBlockSize);
  ASSERT_EQ(tensor->element_type(), PrimitiveType::kUnsigned8);
  ASSERT_EQ(tensor->stride(0), 24);
  ASSERT_EQ(tensor->stride(1), 4);
  ASSERT_EQ(pointer_, tensor->data<uint8_t>().value());
  // Check correct element access
  ir = 0;
  for(int32_t i = 0; i < tensor->shape().dimension(0); i++) {
    ii = i*static_cast<int32_t>(tensor->stride(0))/4;
    for(int32_t j = 0; j < tensor->shape().dimension(1); j++) {
      jj = j*static_cast<int32_t>(tensor->stride(1))/4;
      x = *(tensor->data<uint8_t>().value() + ii + jj);
      ASSERT_EQ(ref[ir++], x);
    }
  }

  // Reshape (C,HxW) to (C,1,HxW)
  tensor->noCopyReshape({channels, 1, height*width});
  // Assert
  ASSERT_EQ(pointer_, tensor->pointer());
  ASSERT_EQ(tensor->shape().dimension(0), channels);
  ASSERT_EQ(tensor->shape().dimension(1), 1);
  ASSERT_EQ(tensor->shape().dimension(2), height*width);
  ASSERT_EQ(tensor->storage_type(), MemoryStorageType::kHost);
  ASSERT_EQ(tensor->bytes_per_element(), 4);
  ASSERT_EQ(tensor->element_count(), kBlockSize);
  ASSERT_EQ(tensor->bytes_size(), 4 * kBlockSize);
  ASSERT_EQ(tensor->rank(), 3);
  ASSERT_EQ(tensor->size(), 4 * kBlockSize);
  ASSERT_EQ(tensor->element_type(), PrimitiveType::kUnsigned8);
  ASSERT_EQ(tensor->stride(0), 24);
  ASSERT_EQ(tensor->stride(1), 24);
  ASSERT_EQ(tensor->stride(2), 4);
  ASSERT_EQ(pointer_, tensor->data<uint8_t>().value());
  // Check correct element access
  ir = 0;
  for(int32_t i = 0; i < tensor->shape().dimension(0); i++) {
    ii = i*static_cast<int32_t>(tensor->stride(0))/4;
    for(int32_t j = 0; j < tensor->shape().dimension(1); j++) {
      jj = j*static_cast<int32_t>(tensor->stride(1))/4;
      for(int32_t k = 0; k < tensor->shape().dimension(2); k++) {
        kk = k*static_cast<int32_t>(tensor->stride(2))/4;
        x = *(tensor->data<uint8_t>().value() + ii + jj + kk);
        ASSERT_EQ(ref[ir++], x);
      }
    }
  }

  // Clean up
  delete tensor;

  ASSERT_TRUE(release_func_params_match);
  release_func_params_match = false;

  ASSERT_EQ(allocator->deinitialize(), GXF_SUCCESS);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(Tensor, insertSingletonDimension) {
  constexpr uint64_t width = 3;
  constexpr uint64_t height = 2;
  constexpr uint64_t channels = 2;
  constexpr uint64_t kBlockSize = channels * width * height;
  constexpr uint64_t kNumBlocks = 1;

  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  constexpr const char* kExtensions[] = {
      "gxf/std/libgxf_std.so",
  };
  const GxfLoadExtensionsInfo info{kExtensions, 1, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &info), GXF_SUCCESS);

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

  // Initialize array data used for testing
  std::array<uint8_t, kBlockSize> arr;
  std::iota(std::begin(arr), std::end(arr), 0);

  // Get pointer to array data
  void* pointer_ = arr.data();
  bool release_func_params_match = false;
  MemoryBuffer::release_function_t release_func = [pointer_, &release_func_params_match](void* pointer) {
    release_func_params_match = (pointer == pointer_);
    return Success;
  };

  // Initialize tensor
  Tensor* tensor = new Tensor();
  auto shape = Shape({width, height, channels});
  tensor->wrapMemory(shape, PrimitiveType::kUnsigned8, 4,
                     Unexpected{GXF_UNINITIALIZED_VALUE}, MemoryStorageType::kHost, pointer_,
                     release_func);
  // Assert
  ASSERT_EQ(pointer_, tensor->pointer());
  ASSERT_EQ(tensor->shape().rank(), 3);
  ASSERT_EQ(tensor->shape().dimension(0), width);
  ASSERT_EQ(tensor->shape().dimension(1), height);
  ASSERT_EQ(tensor->shape().dimension(2), channels);
  ASSERT_EQ(tensor->storage_type(), MemoryStorageType::kHost);
  ASSERT_EQ(tensor->bytes_per_element(), 4);
  ASSERT_EQ(tensor->element_count(), kBlockSize);
  ASSERT_EQ(tensor->bytes_size(), 4 * kBlockSize);
  ASSERT_EQ(tensor->rank(), 3);
  ASSERT_EQ(tensor->size(), 4 * kBlockSize);
  ASSERT_EQ(tensor->element_type(), PrimitiveType::kUnsigned8);
  ASSERT_EQ(tensor->stride(0), 4 * height * channels);
  ASSERT_EQ(tensor->stride(1), 4 * height);
  ASSERT_EQ(tensor->stride(2), 4);
  ASSERT_EQ(pointer_, tensor->data<uint8_t>().value());
  // Check correct element access
  std::array<uint8_t, kBlockSize> ref = {0, 1, 2, 3, 4, 5, 6, 7, 8 ,9, 10, 11};
  int32_t ir = 0, ii = 0, jj = 0, kk = 0, ll = 0, mm = 0, nn = 0;
  uint8_t x = 0;
  for(int32_t i = 0; i < tensor->shape().dimension(0); i++) {
    ii = i*static_cast<int32_t>(tensor->stride(0))/4;
    for(int32_t j = 0; j < tensor->shape().dimension(1); j++) {
      jj = j*static_cast<int32_t>(tensor->stride(1))/4;
      for(int32_t k = 0; k < tensor->shape().dimension(2); k++) {
        kk = k*static_cast<int32_t>(tensor->stride(2))/4;
        x = *(tensor->data<uint8_t>().value() + ii + jj + kk);
        ASSERT_EQ(ref[ir++], x);
      }
    }
  }

  // Insert singleton dimension at front
  tensor->insertSingletonDim(0);
  // Assert
  ASSERT_EQ(pointer_, tensor->pointer());
  ASSERT_EQ(tensor->shape().rank(), 4);
  ASSERT_EQ(tensor->shape().dimension(0), 1);
  ASSERT_EQ(tensor->shape().dimension(1), width);
  ASSERT_EQ(tensor->shape().dimension(2), height);
  ASSERT_EQ(tensor->shape().dimension(3), channels);
  ASSERT_EQ(tensor->storage_type(), MemoryStorageType::kHost);
  ASSERT_EQ(tensor->bytes_per_element(), 4);
  ASSERT_EQ(tensor->element_count(), kBlockSize);
  ASSERT_EQ(tensor->bytes_size(), 4 * kBlockSize);
  ASSERT_EQ(tensor->rank(), 4);
  ASSERT_EQ(tensor->size(), 4 * kBlockSize);
  ASSERT_EQ(tensor->element_type(), PrimitiveType::kUnsigned8);
  ASSERT_EQ(tensor->stride(0), 4 * height * channels * width);
  ASSERT_EQ(tensor->stride(1), 4 * height * channels);
  ASSERT_EQ(tensor->stride(2), 4 * height);
  ASSERT_EQ(tensor->stride(3), 4);
  ASSERT_EQ(pointer_, tensor->data<uint8_t>().value());
  // Check correct element access
  ir = 0;
  for(int32_t i = 0; i < tensor->shape().dimension(0); i++) {
    ii = i*static_cast<int32_t>(tensor->stride(0))/4;
    for(int32_t j = 0; j < tensor->shape().dimension(1); j++) {
      jj = j*static_cast<int32_t>(tensor->stride(1))/4;
      for(int32_t k = 0; k < tensor->shape().dimension(2); k++) {
        kk = k*static_cast<int32_t>(tensor->stride(2))/4;
        for(int32_t l = 0; l < tensor->shape().dimension(3); l++) {
          ll = l*static_cast<int32_t>(tensor->stride(3))/4;
          x = *(tensor->data<uint8_t>().value() + ii + jj + kk + ll);
          ASSERT_EQ(ref[ir++], x);
        }
      }
    }
  }

  // Insert another singleton dimension at front
  tensor->insertSingletonDim(0);
  // Assert
  ASSERT_EQ(pointer_, tensor->pointer());
  ASSERT_EQ(tensor->shape().rank(), 5);
  ASSERT_EQ(tensor->shape().dimension(0), 1);
  ASSERT_EQ(tensor->shape().dimension(1), 1);
  ASSERT_EQ(tensor->shape().dimension(2), width);
  ASSERT_EQ(tensor->shape().dimension(3), height);
  ASSERT_EQ(tensor->shape().dimension(4), channels);
  ASSERT_EQ(tensor->storage_type(), MemoryStorageType::kHost);
  ASSERT_EQ(tensor->bytes_per_element(), 4);
  ASSERT_EQ(tensor->element_count(), kBlockSize);
  ASSERT_EQ(tensor->bytes_size(), 4 * kBlockSize);
  ASSERT_EQ(tensor->rank(), 5);
  ASSERT_EQ(tensor->size(), 4 * kBlockSize);
  ASSERT_EQ(tensor->element_type(), PrimitiveType::kUnsigned8);
  ASSERT_EQ(tensor->stride(0), 4 * height * channels * width);
  ASSERT_EQ(tensor->stride(1), 4 * height * channels * width);
  ASSERT_EQ(tensor->stride(2), 4 * height * channels);
  ASSERT_EQ(tensor->stride(3), 4 * height);
  ASSERT_EQ(tensor->stride(4), 4);
  ASSERT_EQ(pointer_, tensor->data<uint8_t>().value());
  // Check correct element access
  ir = 0;
  for(int32_t i = 0; i < tensor->shape().dimension(0); i++) {
    ii = i*static_cast<int32_t>(tensor->stride(0))/4;
    for(int32_t j = 0; j < tensor->shape().dimension(1); j++) {
      jj = j*static_cast<int32_t>(tensor->stride(1))/4;
      for(int32_t k = 0; k < tensor->shape().dimension(2); k++) {
        kk = k*static_cast<int32_t>(tensor->stride(2))/4;
        for(int32_t l = 0; l < tensor->shape().dimension(3); l++) {
          ll = l*static_cast<int32_t>(tensor->stride(3))/4;
          for(int32_t m = 0; m < tensor->shape().dimension(4); m++) {
            mm = m*static_cast<int32_t>(tensor->stride(4))/4;
            x = *(tensor->data<uint8_t>().value() + ii + jj + kk + ll + mm);
            ASSERT_EQ(ref[ir++], x);
          }
        }
      }
    }
  }

  // Insert another singleton dimension at end
  tensor->insertSingletonDim(5);
  // Assert
  ASSERT_EQ(pointer_, tensor->pointer());
  ASSERT_EQ(tensor->shape().rank(), 6);
  ASSERT_EQ(tensor->shape().dimension(0), 1);
  ASSERT_EQ(tensor->shape().dimension(1), 1);
  ASSERT_EQ(tensor->shape().dimension(2), width);
  ASSERT_EQ(tensor->shape().dimension(3), height);
  ASSERT_EQ(tensor->shape().dimension(4), channels);
  ASSERT_EQ(tensor->shape().dimension(5), 1);
  ASSERT_EQ(tensor->storage_type(), MemoryStorageType::kHost);
  ASSERT_EQ(tensor->bytes_per_element(), 4);
  ASSERT_EQ(tensor->element_count(), kBlockSize);
  ASSERT_EQ(tensor->bytes_size(), 4 * kBlockSize);
  ASSERT_EQ(tensor->rank(), 6);
  ASSERT_EQ(tensor->size(), 4 * kBlockSize);
  ASSERT_EQ(tensor->element_type(), PrimitiveType::kUnsigned8);
  ASSERT_EQ(tensor->stride(0), 4 * height * channels * width);
  ASSERT_EQ(tensor->stride(1), 4 * height * channels * width);
  ASSERT_EQ(tensor->stride(2), 4 * height * channels);
  ASSERT_EQ(tensor->stride(3), 4 * height);
  ASSERT_EQ(tensor->stride(4), 4);
  ASSERT_EQ(tensor->stride(5), 4);
  ASSERT_EQ(pointer_, tensor->data<uint8_t>().value());
  // Check correct element access
  ir = 0;
  for(int32_t i = 0; i < tensor->shape().dimension(0); i++) {
    ii = i*static_cast<int32_t>(tensor->stride(0))/4;
    for(int32_t j = 0; j < tensor->shape().dimension(1); j++) {
      jj = j*static_cast<int32_t>(tensor->stride(1))/4;
      for(int32_t k = 0; k < tensor->shape().dimension(2); k++) {
        kk = k*static_cast<int32_t>(tensor->stride(2))/4;
        for(int32_t l = 0; l < tensor->shape().dimension(3); l++) {
          ll = l*static_cast<int32_t>(tensor->stride(3))/4;
          for(int32_t m = 0; m < tensor->shape().dimension(4); m++) {
            mm = m*static_cast<int32_t>(tensor->stride(4))/4;
            for(int32_t n = 0; n < tensor->shape().dimension(5); n++) {
              nn = n*static_cast<int32_t>(tensor->stride(5))/4;
              x = *(tensor->data<uint8_t>().value() + ii + jj + kk + ll + mm + nn);
              ASSERT_EQ(ref[ir++], x);
            }
          }
        }
      }
    }
  }

  // Clean up
  delete tensor;

  ASSERT_TRUE(release_func_params_match);
  release_func_params_match = false;

  ASSERT_EQ(allocator->deinitialize(), GXF_SUCCESS);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

}  // namespace gxf
}  // namespace nvidia
