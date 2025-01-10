/*
Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "dlpack/dlpack.h"
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"
#include "gxf/std/dlpack_utils.hpp"
#include "gxf/std/tensor.hpp"

namespace nvidia {
namespace gxf {

namespace {

  void checkDLManagedTensorShape(Tensor* tensor) {
    // can also retrieve the DLManagedTensorContext object
    auto maybe_dl_ctx = tensor->toDLManagedTensorContext();
    ASSERT_TRUE(maybe_dl_ctx.has_value());
    auto dl_ctx = maybe_dl_ctx.value();

    // verify that the DLPack structures have the expected shape
    ASSERT_EQ(dl_ctx->tensor.dl_tensor.ndim, tensor->shape().rank());
    for (int i = 0; i < dl_ctx->tensor.dl_tensor.ndim; i++) {
      ASSERT_EQ(dl_ctx->tensor.dl_tensor.shape[i], tensor->shape().dimension(i));
      ASSERT_EQ(dl_ctx->tensor.dl_tensor.strides[i], tensor->stride(i) / tensor->bytes_per_element());
    }
  }

}  // namespace

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

TEST(Tensor, reshapeCustomAllocatorFloat16) {
  // Use of CUDA's __half should work on host for CUDA >= 12.2
  MemoryStorageType storage_type = MemoryStorageType::kSystem;

  constexpr uint64_t num_elements = 512;
  constexpr uint64_t bytes_per_element = 2;
  constexpr uint64_t kBlockSize = num_elements * bytes_per_element;

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
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::UnboundedAllocator", &tid), GXF_SUCCESS);

  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "allocator", &cid), GXF_SUCCESS);

  ASSERT_EQ(GxfParameterSetInt32(context, cid, "storage_type", static_cast<int32_t>(storage_type)),
            GXF_SUCCESS);

  auto maybe_allocator = Handle<Allocator>::Create(context, cid);
  ASSERT_TRUE(maybe_allocator.has_value());

  auto allocator = maybe_allocator.value();

  ASSERT_EQ(allocator->initialize(), GXF_SUCCESS);

  ASSERT_EQ(allocator->is_available_abi(kBlockSize), GXF_SUCCESS);

  Tensor* tensor = new Tensor();

  tensor->reshapeCustom(Shape({1, num_elements}), PrimitiveType::kFloat16, bytes_per_element,
                        Unexpected{GXF_UNINITIALIZED_VALUE}, storage_type, allocator);

  checkDLManagedTensorShape(tensor);

  // verify that __half data can be written to the tensor on the host
  auto maybe_half_ptr = tensor->data<__half>();
  ASSERT_TRUE(maybe_half_ptr.has_value());
  auto half_ptr = maybe_half_ptr.value();
  for (uint64_t i = 0; i < num_elements; i++) {
    half_ptr[i] = __float2half(static_cast<float>(i));
  }
  ASSERT_EQ(__half2float(half_ptr[5]), 5.0f);

  delete tensor;

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

  ASSERT_EQ(GxfParameterSetUInt64(context, cid, "block_size", kBlockSize), GXF_SUCCESS);
  ASSERT_EQ(GxfParameterSetUInt64(context, cid, "num_blocks", kNumBlocks), GXF_SUCCESS);

  auto maybe_allocator = Handle<Allocator>::Create(context, cid);
  ASSERT_TRUE(maybe_allocator.has_value());

  auto allocator = maybe_allocator.value();

  ASSERT_EQ(allocator->initialize(), GXF_SUCCESS);

  ASSERT_EQ(allocator->is_available_abi(kBlockSize), GXF_SUCCESS);

  void* pointer_ = this;
  bool release_func_params_match = false;
  MemoryBuffer::release_function_t release_func = [pointer_,
                                                   &release_func_params_match](void* pointer) {
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
  ASSERT_EQ(tensor->isContiguous().value(), true);
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

  // set the same pointer_ again.
  ASSERT_EQ(tensor->pointer(), pointer_);
  tensor->wrapMemory(Shape({1, kBlockSize}), PrimitiveType::kUnsigned8, 1,
                     Unexpected{GXF_UNINITIALIZED_VALUE}, MemoryStorageType::kHost, pointer_,
                     release_func, false);
  // Calling wrapMemory with the same pointer as the Tensor with `reset_dlpack=false` will have
  // skipped freeBuffer().
  // Given that, the release_func is not called and `release_func_params_match` is still false.
  ASSERT_FALSE(release_func_params_match);
  ASSERT_EQ(allocator->is_available_abi(kBlockSize), GXF_SUCCESS);

  delete tensor;
  ASSERT_TRUE(release_func_params_match);

  ASSERT_EQ(allocator->deinitialize(), GXF_SUCCESS);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

// Test wrapMemory api for PrimitiveType::kCustom
TEST(Tensor, wrapMemoryCustom) {
  constexpr uint64_t kBlockSize = 1024;

  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  constexpr const char* kExtensions[] = {
      "gxf/std/libgxf_std.so",
  };
  const GxfLoadExtensionsInfo info{kExtensions, 1, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &info), GXF_SUCCESS);

  void* pointer_ = this;
  bool release_func_params_match = false;
  MemoryBuffer::release_function_t release_func = [pointer_,
                                                   &release_func_params_match](void* pointer) {
    release_func_params_match = (pointer == pointer_);
    return Success;
  };

  Tensor* tensor = new Tensor();

  tensor->wrapMemory(Shape({kBlockSize}), PrimitiveType::kCustom, sizeof(char),
                     Unexpected{GXF_UNINITIALIZED_VALUE}, MemoryStorageType::kDevice, pointer_,
                     release_func);

  ASSERT_EQ(pointer_, tensor->pointer());
  ASSERT_EQ(tensor->shape().rank(), 1);
  ASSERT_EQ(tensor->shape().dimension(0), kBlockSize);
  ASSERT_EQ(tensor->storage_type(), MemoryStorageType::kDevice);
  ASSERT_EQ(tensor->bytes_per_element(), 1);
  ASSERT_EQ(tensor->element_count(), kBlockSize);
  ASSERT_EQ(tensor->bytes_size(), kBlockSize);
  ASSERT_EQ(tensor->rank(), 1);
  ASSERT_EQ(tensor->size(), kBlockSize);
  ASSERT_EQ(tensor->element_type(), PrimitiveType::kCustom);
  ASSERT_EQ(tensor->stride(0), 1);

  checkDLManagedTensorShape(tensor);

  delete tensor;

  ASSERT_TRUE(release_func_params_match);
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
  MemoryBuffer::release_function_t release_func = [pointer_,
                                                   &release_func_params_match](void* pointer) {
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
  ASSERT_EQ(tensor->isContiguous().value(), true);
  ASSERT_EQ(pointer_, tensor->data<uint8_t>().value());
  // Check correct element access
  std::array<uint8_t, kBlockSize> ref = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  int32_t ir = 0, ii = 0, jj = 0, kk = 0;
  uint8_t x = 0;
  for (int32_t i = 0; i < tensor->shape().dimension(0); i++) {
    ii = i*static_cast<int32_t>(tensor->stride(0))/4;
    for (int32_t j = 0; j < tensor->shape().dimension(1); j++) {
      jj = j*static_cast<int32_t>(tensor->stride(1))/4;
      for (int32_t k = 0; k < tensor->shape().dimension(2); k++) {
        kk = k*static_cast<int32_t>(tensor->stride(2))/4;
        x = *(tensor->data<uint8_t>().value() + ii + jj + kk);
        ASSERT_EQ(ref[ir++], x);
      }
    }
  }

  checkDLManagedTensorShape(tensor);

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
  ASSERT_EQ(tensor->isContiguous().value(), false);
  ASSERT_EQ(pointer_, tensor->data<uint8_t>().value());
  // Check correct element access
  ref = {0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11};
  ir = 0;
  for (int32_t i = 0; i < tensor->shape().dimension(0); i++) {
    ii = i*static_cast<int32_t>(tensor->stride(0))/4;
    for (int32_t j = 0; j < tensor->shape().dimension(1); j++) {
      jj = j*static_cast<int32_t>(tensor->stride(1))/4;
      for (int32_t k = 0; k < tensor->shape().dimension(2); k++) {
        kk = k*static_cast<int32_t>(tensor->stride(2))/4;
        x = *(tensor->data<uint8_t>().value() + ii + jj + kk);
        ASSERT_EQ(ref[ir++], x);
      }
    }
  }

  // verify that the DLManagedTensorContext object has the new shape
  checkDLManagedTensorShape(tensor);

  // Clean up
  delete tensor;

  ASSERT_TRUE(release_func_params_match);
  release_func_params_match = false;

  ASSERT_EQ(allocator->deinitialize(), GXF_SUCCESS);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(Tensor, noCopyReshape) {
  constexpr uint64_t width = 5;
  constexpr uint64_t height = 4;
  constexpr uint64_t channels = 2;
  constexpr uint64_t kBlockSize = channels * width * height;
  constexpr uint64_t kNumBlocks = 1;
  constexpr uint64_t bytes_per_element = 4;

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
  MemoryBuffer::release_function_t release_func = [pointer_,
                                                   &release_func_params_match](void* pointer) {
    release_func_params_match = (pointer == pointer_);
    return Success;
  };

  // Initialize tensor
  Tensor* tensor = new Tensor();
  auto shape = Shape({width, height, channels});

  tensor->wrapMemory(shape, PrimitiveType::kUnsigned8, bytes_per_element,
                     Unexpected{GXF_UNINITIALIZED_VALUE}, MemoryStorageType::kHost, pointer_,
                     release_func);

  // Assert
  ASSERT_EQ(pointer_, tensor->pointer());
  ASSERT_EQ(tensor->shape().rank(), 3);
  ASSERT_EQ(tensor->shape().dimension(0), width);
  ASSERT_EQ(tensor->shape().dimension(1), height);
  ASSERT_EQ(tensor->shape().dimension(2), channels);
  ASSERT_EQ(tensor->storage_type(), MemoryStorageType::kHost);
  ASSERT_EQ(tensor->bytes_per_element(), bytes_per_element);
  ASSERT_EQ(tensor->element_count(), kBlockSize);
  ASSERT_EQ(tensor->bytes_size(), bytes_per_element * kBlockSize);
  ASSERT_EQ(tensor->rank(), 3);
  ASSERT_EQ(tensor->size(), bytes_per_element * kBlockSize);
  ASSERT_EQ(tensor->element_type(), PrimitiveType::kUnsigned8);
  ASSERT_EQ(tensor->stride(0), bytes_per_element * height * channels);
  ASSERT_EQ(tensor->stride(1), bytes_per_element * channels);
  ASSERT_EQ(tensor->stride(2), bytes_per_element);
  ASSERT_EQ(tensor->isContiguous().value(), true);
  ASSERT_EQ(pointer_, tensor->data<uint8_t>().value());
  // Check correct element access

  std::array<uint8_t, kBlockSize> ref;
  for (uint64_t i = 0; i < width * height * channels; i++) {
    ref[i] = i;
  }
  int32_t ir = 0, ii = 0, jj = 0, kk = 0;
  uint8_t x = 0;
  for (int32_t i = 0; i < tensor->shape().dimension(0); i++) {
    ii = i*static_cast<int32_t>(tensor->stride(0))/bytes_per_element;
    for (int32_t j = 0; j < tensor->shape().dimension(1); j++) {
      jj = j*static_cast<int32_t>(tensor->stride(1))/bytes_per_element;
      for (int32_t k = 0; k < tensor->shape().dimension(2); k++) {
        kk = k*static_cast<int32_t>(tensor->stride(2))/bytes_per_element;
        x = *(tensor->data<uint8_t>().value() + ii + jj + kk);
        ASSERT_EQ(ref[ir++], x);
      }
    }
  }
  checkDLManagedTensorShape(tensor);

  // Reshape (W,H,C) to (C,H,W)
  tensor->noCopyReshape({channels, height, width});
  // Assert
  ASSERT_EQ(pointer_, tensor->pointer());
  ASSERT_EQ(tensor->shape().dimension(0), channels);
  ASSERT_EQ(tensor->shape().dimension(1), height);
  ASSERT_EQ(tensor->shape().dimension(2), width);
  ASSERT_EQ(tensor->storage_type(), MemoryStorageType::kHost);
  ASSERT_EQ(tensor->bytes_per_element(), bytes_per_element);
  ASSERT_EQ(tensor->element_count(), kBlockSize);
  ASSERT_EQ(tensor->bytes_size(), bytes_per_element * kBlockSize);
  ASSERT_EQ(tensor->rank(), 3);
  ASSERT_EQ(tensor->size(), bytes_per_element * kBlockSize);
  ASSERT_EQ(tensor->element_type(), PrimitiveType::kUnsigned8);
  ASSERT_EQ(tensor->stride(0), bytes_per_element * width * height);
  ASSERT_EQ(tensor->stride(1), bytes_per_element * width);
  ASSERT_EQ(tensor->stride(2), bytes_per_element);
  ASSERT_EQ(tensor->isContiguous().value(), true);
  ASSERT_EQ(pointer_, tensor->data<uint8_t>().value());
  // Check correct element access
  ir = 0;
  for (int32_t i = 0; i < tensor->shape().dimension(0); i++) {
    ii = i*static_cast<int32_t>(tensor->stride(0))/bytes_per_element;
    for (int32_t j = 0; j < tensor->shape().dimension(1); j++) {
      jj = j*static_cast<int32_t>(tensor->stride(1))/bytes_per_element;
      for (int32_t k = 0; k < tensor->shape().dimension(2); k++) {
        kk = k*static_cast<int32_t>(tensor->stride(2))/bytes_per_element;
        x = *(tensor->data<uint8_t>().value() + ii + jj + kk);
        ASSERT_EQ(ref[ir++], x);
      }
    }
  }
  checkDLManagedTensorShape(tensor);

  // Reshape (C,H,W) to (C,HxW)
  tensor->noCopyReshape({channels, height*width});
  // Assert
  ASSERT_EQ(pointer_, tensor->pointer());
  ASSERT_EQ(tensor->shape().dimension(0), channels);
  ASSERT_EQ(tensor->shape().dimension(1), height*width);
  ASSERT_EQ(tensor->storage_type(), MemoryStorageType::kHost);
  ASSERT_EQ(tensor->bytes_per_element(), bytes_per_element);
  ASSERT_EQ(tensor->element_count(), kBlockSize);
  ASSERT_EQ(tensor->bytes_size(), bytes_per_element * kBlockSize);
  ASSERT_EQ(tensor->rank(), 2);
  ASSERT_EQ(tensor->size(), bytes_per_element * kBlockSize);
  ASSERT_EQ(tensor->element_type(), PrimitiveType::kUnsigned8);
  ASSERT_EQ(tensor->stride(0), bytes_per_element * width * height);
  ASSERT_EQ(tensor->stride(1), bytes_per_element);
  ASSERT_EQ(tensor->isContiguous().value(), true);
  ASSERT_EQ(pointer_, tensor->data<uint8_t>().value());
  // Check correct element access
  ir = 0;
  for (int32_t i = 0; i < tensor->shape().dimension(0); i++) {
    ii = i*static_cast<int32_t>(tensor->stride(0))/bytes_per_element;
    for (int32_t j = 0; j < tensor->shape().dimension(1); j++) {
      jj = j*static_cast<int32_t>(tensor->stride(1))/bytes_per_element;
      x = *(tensor->data<uint8_t>().value() + ii + jj);
      ASSERT_EQ(ref[ir++], x);
    }
  }
  checkDLManagedTensorShape(tensor);

  // Reshape (C,HxW) to (C,1,HxW)
  tensor->noCopyReshape({channels, 1, height*width});
  // Assert
  ASSERT_EQ(pointer_, tensor->pointer());
  ASSERT_EQ(tensor->shape().dimension(0), channels);
  ASSERT_EQ(tensor->shape().dimension(1), 1);
  ASSERT_EQ(tensor->shape().dimension(2), height*width);
  ASSERT_EQ(tensor->storage_type(), MemoryStorageType::kHost);
  ASSERT_EQ(tensor->bytes_per_element(), bytes_per_element);
  ASSERT_EQ(tensor->element_count(), kBlockSize);
  ASSERT_EQ(tensor->bytes_size(), bytes_per_element * kBlockSize);
  ASSERT_EQ(tensor->rank(), 3);
  ASSERT_EQ(tensor->size(), bytes_per_element * kBlockSize);
  ASSERT_EQ(tensor->element_type(), PrimitiveType::kUnsigned8);
  ASSERT_EQ(tensor->stride(0), bytes_per_element * width * height);
  ASSERT_EQ(tensor->stride(1), bytes_per_element * width * height);
  ASSERT_EQ(tensor->stride(2), bytes_per_element);
  ASSERT_EQ(tensor->isContiguous().value(), true);
  ASSERT_EQ(pointer_, tensor->data<uint8_t>().value());
  // Check correct element access
  ir = 0;
  for (int32_t i = 0; i < tensor->shape().dimension(0); i++) {
    ii = i*static_cast<int32_t>(tensor->stride(0))/bytes_per_element;
    for (int32_t j = 0; j < tensor->shape().dimension(1); j++) {
      jj = j*static_cast<int32_t>(tensor->stride(1))/bytes_per_element;
      for (int32_t k = 0; k < tensor->shape().dimension(2); k++) {
        kk = k*static_cast<int32_t>(tensor->stride(2))/bytes_per_element;
        x = *(tensor->data<uint8_t>().value() + ii + jj + kk);
        ASSERT_EQ(ref[ir++], x);
      }
    }
  }
  checkDLManagedTensorShape(tensor);

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
  MemoryBuffer::release_function_t release_func = [pointer_,
                                                   &release_func_params_match](void* pointer) {
    release_func_params_match = (pointer == pointer_);
    return Success;
  };

  // Initialize tensor
  Tensor* tensor = new Tensor();
  auto shape = Shape({width, height, channels});
  tensor->wrapMemory(shape, PrimitiveType::kUnsigned8, 4, Unexpected{GXF_UNINITIALIZED_VALUE},
                     MemoryStorageType::kHost, pointer_, release_func);
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
  ASSERT_EQ(tensor->isContiguous().value(), true);
  ASSERT_EQ(pointer_, tensor->data<uint8_t>().value());
  // Check correct element access
  std::array<uint8_t, kBlockSize> ref = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  int32_t ir = 0, ii = 0, jj = 0, kk = 0, ll = 0, mm = 0, nn = 0;
  uint8_t x = 0;
  for (int32_t i = 0; i < tensor->shape().dimension(0); i++) {
    ii = i*static_cast<int32_t>(tensor->stride(0))/4;
    for (int32_t j = 0; j < tensor->shape().dimension(1); j++) {
      jj = j*static_cast<int32_t>(tensor->stride(1))/4;
      for (int32_t k = 0; k < tensor->shape().dimension(2); k++) {
        kk = k*static_cast<int32_t>(tensor->stride(2))/4;
        x = *(tensor->data<uint8_t>().value() + ii + jj + kk);
        ASSERT_EQ(ref[ir++], x);
      }
    }
  }

  checkDLManagedTensorShape(tensor);

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
  ASSERT_EQ(tensor->isContiguous().value(), true);
  ASSERT_EQ(pointer_, tensor->data<uint8_t>().value());
  // Check correct element access
  ir = 0;
  for (int32_t i = 0; i < tensor->shape().dimension(0); i++) {
    ii = i*static_cast<int32_t>(tensor->stride(0))/4;
    for (int32_t j = 0; j < tensor->shape().dimension(1); j++) {
      jj = j*static_cast<int32_t>(tensor->stride(1))/4;
      for (int32_t k = 0; k < tensor->shape().dimension(2); k++) {
        kk = k*static_cast<int32_t>(tensor->stride(2))/4;
        for (int32_t l = 0; l < tensor->shape().dimension(3); l++) {
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
  ASSERT_EQ(tensor->isContiguous().value(), true);
  ASSERT_EQ(pointer_, tensor->data<uint8_t>().value());
  // Check correct element access
  ir = 0;
  for (int32_t i = 0; i < tensor->shape().dimension(0); i++) {
    ii = i*static_cast<int32_t>(tensor->stride(0))/4;
    for (int32_t j = 0; j < tensor->shape().dimension(1); j++) {
      jj = j*static_cast<int32_t>(tensor->stride(1))/4;
      for (int32_t k = 0; k < tensor->shape().dimension(2); k++) {
        kk = k*static_cast<int32_t>(tensor->stride(2))/4;
        for (int32_t l = 0; l < tensor->shape().dimension(3); l++) {
          ll = l*static_cast<int32_t>(tensor->stride(3))/4;
          for (int32_t m = 0; m < tensor->shape().dimension(4); m++) {
            mm = m*static_cast<int32_t>(tensor->stride(4))/4;
            x = *(tensor->data<uint8_t>().value() + ii + jj + kk + ll + mm);
            ASSERT_EQ(ref[ir++], x);
          }
        }
      }
    }
  }
  checkDLManagedTensorShape(tensor);

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
  ASSERT_EQ(tensor->isContiguous().value(), true);
  ASSERT_EQ(pointer_, tensor->data<uint8_t>().value());
  // Check correct element access
  ir = 0;
  for (int32_t i = 0; i < tensor->shape().dimension(0); i++) {
    ii = i*static_cast<int32_t>(tensor->stride(0))/4;
    for (int32_t j = 0; j < tensor->shape().dimension(1); j++) {
      jj = j*static_cast<int32_t>(tensor->stride(1))/4;
      for (int32_t k = 0; k < tensor->shape().dimension(2); k++) {
        kk = k*static_cast<int32_t>(tensor->stride(2))/4;
        for (int32_t l = 0; l < tensor->shape().dimension(3); l++) {
          ll = l*static_cast<int32_t>(tensor->stride(3))/4;
          for (int32_t m = 0; m < tensor->shape().dimension(4); m++) {
            mm = m*static_cast<int32_t>(tensor->stride(4))/4;
            for (int32_t n = 0; n < tensor->shape().dimension(5); n++) {
              nn = n*static_cast<int32_t>(tensor->stride(5))/4;
              x = *(tensor->data<uint8_t>().value() + ii + jj + kk + ll + mm + nn);
              ASSERT_EQ(ref[ir++], x);
            }
          }
        }
      }
    }
  }
  checkDLManagedTensorShape(tensor);

  // Clean up
  delete tensor;

  ASSERT_TRUE(release_func_params_match);
  release_func_params_match = false;

  ASSERT_EQ(allocator->deinitialize(), GXF_SUCCESS);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

using StorageParamPair = std::pair<MemoryStorageType, DLDevice>;

class TensorStorageParameterizedTestFixture : public ::testing::TestWithParam<StorageParamPair> {};

INSTANTIATE_TEST_CASE_P(
    Tensor, TensorStorageParameterizedTestFixture,
    ::testing::Values(StorageParamPair{MemoryStorageType::kDevice,
                                       DLDevice{.device_type = kDLCUDA, .device_id = 0}},
                      StorageParamPair{MemoryStorageType::kHost,
                                       DLDevice{.device_type = kDLCUDAHost, .device_id = 0}},
                      StorageParamPair{MemoryStorageType::kSystem,
                                       DLDevice{.device_type = kDLCPU, .device_id = 0}}));

TEST_P(TensorStorageParameterizedTestFixture, dldeviceFromPointer) {
  auto [storage_type, expected_dldevice] = GetParam();

  gxf_context_t context;
  constexpr size_t kBlockSizeSmall = 100;
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
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::UnboundedAllocator", &tid), GXF_SUCCESS);
  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "allocator", &cid), GXF_SUCCESS);
  ASSERT_EQ(GxfParameterSetInt32(context, cid, "storage_type", static_cast<int32_t>(storage_type)),
            GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);
  Allocator* allocator = static_cast<Allocator*>(pointer);

  ASSERT_EQ(allocator->initialize(), GXF_SUCCESS);

  ASSERT_TRUE(allocator->is_available(kBlockSizeSmall));
  auto block1 = allocator->allocate(kBlockSizeSmall, storage_type);
  ASSERT_TRUE(block1.has_value());
  auto data_pointer = block1.value();

  MemoryBuffer::release_function_t release_func = [](void* pointer) { return Success; };

  // Initialize tensor
  Tensor* tensor = new Tensor();
  auto shape = Shape({kBlockSizeSmall, 1});
  tensor->wrapMemory(shape, PrimitiveType::kUnsigned8, 1, Unexpected{GXF_UNINITIALIZED_VALUE},
                     storage_type, data_pointer, release_func);

  // can retrieve the DLPack device type from the tensor's pointer method
  auto maybe_dldevice = DLDeviceFromPointer(tensor->pointer());
  ASSERT_TRUE(maybe_dldevice.has_value());
  DLDevice dldevice = maybe_dldevice.value();
  ASSERT_EQ(dldevice.device_type, expected_dldevice.device_type);
  ASSERT_EQ(dldevice.device_id, expected_dldevice.device_id);

  // Clean up
  delete tensor;
  ASSERT_EQ(allocator->deinitialize(), GXF_SUCCESS);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

using DataTypePair = std::tuple<PrimitiveType, DLDataType>;

class TensorDataTypeParameterizedTestFixture : public ::testing::TestWithParam<DataTypePair> {};

INSTANTIATE_TEST_CASE_P(
    Tensor, TensorDataTypeParameterizedTestFixture,
    ::testing::Values(
        DataTypePair{PrimitiveType::kUnsigned8, DLDataType{.code = kDLUInt, .bits = 8, .lanes = 1}},
        DataTypePair{PrimitiveType::kUnsigned16,
                     DLDataType{.code = kDLUInt, .bits = 16, .lanes = 1}},
        DataTypePair{PrimitiveType::kUnsigned32,
                     DLDataType{.code = kDLUInt, .bits = 32, .lanes = 1}},
        DataTypePair{PrimitiveType::kUnsigned64,
                     DLDataType{.code = kDLUInt, .bits = 64, .lanes = 1}},
        DataTypePair{PrimitiveType::kInt8, DLDataType{.code = kDLInt, .bits = 8, .lanes = 1}},
        DataTypePair{PrimitiveType::kInt16, DLDataType{.code = kDLInt, .bits = 16, .lanes = 1}},
        DataTypePair{PrimitiveType::kInt32, DLDataType{.code = kDLInt, .bits = 32, .lanes = 1}},
        DataTypePair{PrimitiveType::kInt64, DLDataType{.code = kDLInt, .bits = 64, .lanes = 1}},
        DataTypePair{PrimitiveType::kFloat32, DLDataType{.code = kDLFloat, .bits = 32, .lanes = 1}},
        DataTypePair{PrimitiveType::kFloat64, DLDataType{.code = kDLFloat, .bits = 64, .lanes = 1}},
        DataTypePair{PrimitiveType::kComplex64,
                     DLDataType{.code = kDLComplex, .bits = 64, .lanes = 1}},
        DataTypePair{PrimitiveType::kComplex128,
                     DLDataType{.code = kDLComplex, .bits = 128, .lanes = 1}},
        // Adding this case only to test the conversion of kCustom -> kDLOpaqueHandle
        // The bits value has to be ignored since bytes_per_element of kCustom is not fixed / known
        DataTypePair{PrimitiveType::kCustom,
                     DLDataType{.code = kDLOpaqueHandle, .bits = 0, .lanes = 1}}));

TEST_P(TensorDataTypeParameterizedTestFixture, primitiveTypeFromDLDataType) {
  auto [true_primitive_type, dldata_type] = GetParam();

  auto maybe_primitive = PrimitiveTypeFromDLDataType(dldata_type);
  ASSERT_TRUE(maybe_primitive.has_value());
  EXPECT_EQ(maybe_primitive.value(), true_primitive_type);
}

TEST_P(TensorDataTypeParameterizedTestFixture, primitiveTypeToDLDataType) {
  auto [primitive_type, true_dldata_type] = GetParam();

  auto maybe_dldatatype = PrimitiveTypeToDLDataType(primitive_type);
  ASSERT_TRUE(maybe_dldatatype.has_value());
  DLDataType dldatatype = maybe_dldatatype.value();
  EXPECT_EQ(dldatatype.code, true_dldata_type.code);
  EXPECT_EQ(dldatatype.bits, true_dldata_type.bits);
  EXPECT_EQ(dldatatype.lanes, true_dldata_type.lanes);
}

class TensorInvalidDataParameterizedTestFixture : public ::testing::TestWithParam<DLDataType> {};

INSTANTIATE_TEST_CASE_P(Tensor, TensorInvalidDataParameterizedTestFixture,
                        ::testing::Values(
                            // unsupported bit depths for supported codes
                            DLDataType{.code = kDLFloat, .bits = 128, .lanes = 1},
                            DLDataType{.code = kDLUInt, .bits = 128, .lanes = 1},
                            DLDataType{.code = kDLInt, .bits = 128, .lanes = 1},
                            DLDataType{.code = kDLComplex, .bits = 32, .lanes = 1},
                            // unsupported codes
                            DLDataType{.code = kDLBool, .bits = 8, .lanes = 1},
                            DLDataType{.code = kDLBfloat, .bits = 16, .lanes = 1}));

TEST_P(TensorInvalidDataParameterizedTestFixture, primitiveTypeFromUnsupportedDLDataType) {
  DLDataType dldata_type = GetParam();

  auto maybe_primitive = PrimitiveTypeFromDLDataType(dldata_type);
  ASSERT_FALSE(maybe_primitive.has_value());
  EXPECT_EQ(maybe_primitive.error(), GXF_INVALID_DATA_FORMAT);
}

TEST(Tensor, shapeFromDLTensor) {
  std::vector<int64_t> data_shape{16, 32, 4};

  DLTensor dltensor;
  dltensor.ndim = data_shape.size();
  dltensor.shape = data_shape.data();

  auto maybe_shape = ShapeFromDLTensor(&dltensor);
  EXPECT_TRUE(maybe_shape.has_value());
  Shape shape = maybe_shape.value();
  EXPECT_EQ(shape.rank(), dltensor.ndim);
  EXPECT_EQ(shape.dimension(0), data_shape[0]);
  EXPECT_EQ(shape.dimension(1), data_shape[1]);
  EXPECT_EQ(shape.dimension(2), data_shape[2]);

  // will return Unexpected in case of ndim > Shape::kMaxRank
  int large_ndim = Shape::kMaxRank + 1;
  data_shape.reserve(large_ndim);
  dltensor.ndim = large_ndim;
  maybe_shape = ShapeFromDLTensor(&dltensor);
  EXPECT_FALSE(maybe_shape.has_value());
  EXPECT_EQ(maybe_shape.error(), GXF_INVALID_DATA_FORMAT);
}

TEST(Tensor, stridesFromDLTensorDefaultStrides) {
  std::vector<int64_t> data_shape{16, 32, 4};

  DLTensor dltensor;
  dltensor.ndim = data_shape.size();
  dltensor.shape = data_shape.data();
  dltensor.strides = nullptr;  // default is row-major (C-contiguous)
  dltensor.dtype.bits = 8;

  auto maybe_strides = StridesFromDLTensor(&dltensor);
  EXPECT_TRUE(maybe_strides.has_value());
  auto strides = maybe_strides.value();
  size_t ndim = dltensor.ndim;
  EXPECT_EQ(strides[ndim - 1], dltensor.dtype.bits / 8);
  EXPECT_EQ(strides[ndim - 2], strides[ndim - 1] * data_shape[ndim - 1]);
  EXPECT_EQ(strides[ndim - 3], strides[ndim - 2] * data_shape[ndim - 2]);

  // will return Unexpected in case of ndim > Shape::kMaxRank
  int large_ndim = Shape::kMaxRank + 1;
  data_shape.reserve(large_ndim);
  dltensor.ndim = large_ndim;
  maybe_strides = StridesFromDLTensor(&dltensor);
  EXPECT_FALSE(maybe_strides.has_value());
  EXPECT_EQ(maybe_strides.error(), GXF_INVALID_DATA_FORMAT);
}

TEST(Tensor, stridesFromDLTensorExplicitStrides) {
  std::vector<int64_t> data_shape{16, 32, 4};
  std::vector<int64_t> data_strides{1, 4, 128};  // column-major strides

  DLTensor dltensor;
  dltensor.ndim = data_shape.size();
  dltensor.shape = data_shape.data();
  dltensor.strides = data_strides.data();
  dltensor.dtype.bits = 8;

  auto maybe_strides = StridesFromDLTensor(&dltensor);
  EXPECT_TRUE(maybe_strides.has_value());
  auto strides = maybe_strides.value();
  EXPECT_EQ(strides[0], data_strides[0]);
  EXPECT_EQ(strides[1], data_strides[1]);
  EXPECT_EQ(strides[2], data_strides[2]);
}

TEST(Tensor, toDLManagedTensorContext) {
  MemoryStorageType storage_type = MemoryStorageType::kSystem;
  DLDeviceType expected_device_type = kDLCPU;

  PrimitiveType element_type = PrimitiveType::kUnsigned16;
  DLDataTypeCode expected_dtype_code = kDLUInt;

  auto bytes_per_element = PrimitiveTypeSize(element_type);
  gxf_context_t context;
  constexpr size_t kBlockSizeSmall = 100;
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
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::UnboundedAllocator", &tid), GXF_SUCCESS);
  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "allocator", &cid), GXF_SUCCESS);
  ASSERT_EQ(GxfParameterSetInt32(context, cid, "storage_type", static_cast<int32_t>(storage_type)),
            GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);
  Allocator* allocator = static_cast<Allocator*>(pointer);

  ASSERT_EQ(allocator->initialize(), GXF_SUCCESS);

  ASSERT_TRUE(allocator->is_available(kBlockSizeSmall));
  int rows = 16, cols = 8;
  auto block1 =
      allocator->allocate(kBlockSizeSmall * rows * cols * bytes_per_element, storage_type);
  ASSERT_TRUE(block1.has_value());

  //Query block size of this allocator. Defaults to 1 for byte-based allocators.
  auto blockSize = allocator->block_size_abi();
  ASSERT_EQ(blockSize, 1);

  auto data_pointer = block1.value();

  MemoryBuffer::release_function_t release_func = [](void* pointer) { return Success; };

  // Initialize tensor
  Tensor* tensor = new Tensor();
  auto shape = Shape({kBlockSizeSmall, rows, cols});
  tensor->wrapMemory(shape, element_type, bytes_per_element, Unexpected{GXF_UNINITIALIZED_VALUE},
                     storage_type, data_pointer, release_func);
  ASSERT_EQ(tensor->bytes_per_element(), bytes_per_element);
  ASSERT_EQ(tensor->element_type(), element_type);

  // can also retrieve the DLManagedTensorContext object
  auto maybe_dl_ctx = tensor->toDLManagedTensorContext();
  ASSERT_TRUE(maybe_dl_ctx.has_value());
  auto dl_ctx = maybe_dl_ctx.value();
  // dl_ctx is also stored as a class member so should have count 2
  ASSERT_EQ(dl_ctx.use_count(), 2);

  // verify that the DLPack structures have the expected values
  auto dl_managed_tensor = dl_ctx->tensor;
  DLTensor& dl_tensor = dl_managed_tensor.dl_tensor;
  ASSERT_EQ(dl_tensor.data, tensor->pointer());
  ASSERT_EQ(dl_tensor.device.device_type, expected_device_type);
  ASSERT_EQ(dl_tensor.device.device_id, 0);
  ASSERT_EQ(dl_tensor.ndim, tensor->shape().rank());
  ASSERT_EQ(dl_tensor.dtype.code, expected_dtype_code);
  ASSERT_EQ(dl_tensor.dtype.bits, 8 * tensor->bytes_per_element());
  ASSERT_EQ(dl_tensor.dtype.lanes, 1);
  for (int i = 0; i < dl_tensor.ndim; i++) {
    ASSERT_EQ(dl_tensor.shape[i], tensor->shape().dimension(i));
    ASSERT_EQ(dl_tensor.strides[i], tensor->stride(i) / tensor->bytes_per_element());
  }
  ASSERT_EQ(dl_tensor.byte_offset, 0);

  // verify that a shaped memory buffer was stored as the memory_ref member of dl_ctx
  auto shaped_buffer = static_cast<MemoryBuffer*>(dl_ctx->memory_ref.get());
  ASSERT_EQ(shaped_buffer->pointer(), tensor->pointer());

  // verify that stored shape and strides are as expected
  for (int i = 0; i < dl_tensor.ndim; i++) {
    ASSERT_EQ(dl_ctx->dl_shape[i], tensor->shape().dimension(i));
    ASSERT_EQ(dl_ctx->dl_strides[i], tensor->stride(i) / tensor->bytes_per_element());
  }

  // can also retrieve a raw DLManagedTensor pointer via toDLPack
  auto maybe_dlpack_managed_tensor_ptr = tensor->toDLPack();
  ASSERT_TRUE(maybe_dlpack_managed_tensor_ptr.has_value());
  DLManagedTensor* dlpack_managed_tensor_ptr = maybe_dlpack_managed_tensor_ptr.value();
  DLTensor& dlpack_tensor = dlpack_managed_tensor_ptr->dl_tensor;
  // verify that the DLPack structures have the expected values
  ASSERT_EQ(dlpack_tensor.data, tensor->pointer());
  ASSERT_EQ(dlpack_tensor.device.device_type, expected_device_type);
  ASSERT_EQ(dlpack_tensor.device.device_id, 0);
  ASSERT_EQ(dlpack_tensor.ndim, tensor->shape().rank());
  ASSERT_EQ(dlpack_tensor.dtype.code, expected_dtype_code);
  ASSERT_EQ(dlpack_tensor.dtype.bits, 8 * tensor->bytes_per_element());
  ASSERT_EQ(dlpack_tensor.dtype.lanes, 1);
  for (int i = 0; i < dl_tensor.ndim; i++) {
    ASSERT_EQ(dlpack_tensor.shape[i], tensor->shape().dimension(i));
    ASSERT_EQ(dlpack_tensor.strides[i], tensor->stride(i) / tensor->bytes_per_element());
  }
  ASSERT_EQ(dlpack_tensor.byte_offset, 0);

  // can call delete on the DLManagedTensor*
  ASSERT_NE(dlpack_managed_tensor_ptr->deleter, nullptr);
  dlpack_managed_tensor_ptr->deleter(dlpack_managed_tensor_ptr);
  dlpack_managed_tensor_ptr->deleter = nullptr;

  // Clean up
  delete tensor;
  ASSERT_EQ(allocator->deinitialize(), GXF_SUCCESS);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

class TensorFromDLPackBooleanPairFixture : public ::testing::TestWithParam<std::pair<bool, bool>> {
};

INSTANTIATE_TEST_CASE_P(Tensor, TensorFromDLPackBooleanPairFixture,
                        ::testing::Values(std::make_pair(false, false), std::make_pair(false, true),
                                          std::make_pair(true, false), std::make_pair(true, true)));

TEST_P(TensorFromDLPackBooleanPairFixture, fromDLPack) {
  auto [use_shared_ptr_api, use_constructor] = GetParam();

  std::vector<int64_t> shape{32, 16};
  MemoryStorageType storage_type = MemoryStorageType::kSystem;

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
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::UnboundedAllocator", &tid), GXF_SUCCESS);
  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "allocator", &cid), GXF_SUCCESS);
  ASSERT_EQ(GxfParameterSetInt32(context, cid, "storage_type", static_cast<int32_t>(storage_type)),
            GXF_SUCCESS);

  auto maybe_allocator = Handle<Allocator>::Create(context, cid);
  ASSERT_TRUE(maybe_allocator.has_value());

  auto allocator = maybe_allocator.value();

  ASSERT_EQ(allocator->initialize(), GXF_SUCCESS);

  void* data_ptr = ::operator new(shape[0] * shape[1] * sizeof(float));
  DLManagedTensor dl_managed_tensor;
  DLTensor& dl_tensor = dl_managed_tensor.dl_tensor;
  dl_tensor.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
  dl_tensor.dtype = DLDataType{.code = kDLFloat, .bits = 32, .lanes = 1};
  dl_tensor.ndim = 2;
  dl_tensor.strides = nullptr;
  dl_tensor.shape = shape.data();
  dl_tensor.byte_offset = 0;
  dl_tensor.data = data_ptr;

  dl_managed_tensor.manager_ctx = nullptr;
  dl_managed_tensor.deleter = [](struct DLManagedTensor* self) {
    GXF_LOG_INFO("dl_managed_tensor.deleter called");
    ::operator delete(self->dl_tensor.data);
    self->dl_tensor.data = nullptr;
  };

  Tensor* tensor = nullptr;
  if (use_shared_ptr_api) {
    // zero-copy Tensor initialization from std::shared_ptr<DLManagedTensorContext>
    auto dl_ctx = std::make_shared<DLManagedTensorContext>();
    dl_ctx->memory_ref = std::make_shared<DLManagedMemoryBuffer>(&dl_managed_tensor);
    dl_ctx->tensor = dl_managed_tensor;

    if (use_constructor) {
      tensor = new Tensor(dl_ctx);
    } else {
      tensor = new Tensor();
      tensor->fromDLPack(dl_ctx);
    }
  } else {
    // zero-copy Tensor initialization from DLManagedTensor structure pointer
    if (use_constructor) {
      tensor = new Tensor(&dl_managed_tensor);
    } else {
      tensor = new Tensor();
      tensor->fromDLPack(&dl_managed_tensor);
    }
  }

  // verify expected tensor properties
  ASSERT_EQ(data_ptr, tensor->pointer());
  ASSERT_EQ(tensor->shape().rank(), 2);
  ASSERT_EQ(tensor->shape().dimension(0), 32);
  ASSERT_EQ(tensor->shape().dimension(1), 16);
  ASSERT_EQ(tensor->stride(1), sizeof(float));
  ASSERT_EQ(tensor->stride(0), sizeof(float) * shape[1]);
  ASSERT_EQ(tensor->storage_type(), storage_type);
  ASSERT_EQ(tensor->bytes_per_element(), sizeof(float));
  ASSERT_EQ(tensor->element_count(), shape[0] * shape[1]);
  ASSERT_EQ(tensor->bytes_size(), sizeof(float) * shape[0] * shape[1]);
  ASSERT_EQ(tensor->element_type(), PrimitiveType::kFloat32);

  // Tensor deletion should will have set dl_tensor.data to nullptr
  ASSERT_NE(dl_tensor.data, nullptr);
  delete tensor;
  ASSERT_EQ(dl_tensor.data, nullptr);

  // Clean up
  ASSERT_EQ(allocator->deinitialize(), GXF_SUCCESS);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

}  // namespace gxf
}  // namespace nvidia
