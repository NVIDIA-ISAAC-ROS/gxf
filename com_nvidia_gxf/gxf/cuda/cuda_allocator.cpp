/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "gxf/cuda/cuda_allocator.hpp"
#include "gxf/std/allocator.hpp"

namespace nvidia {
namespace gxf {
Expected<byte*> CudaAllocator::allocate_async(uint64_t size, cudaStream_t stream) {
  void* result;
  const gxf_result_t code = allocate_async_abi(size, &result, stream);
  return ExpectedOrCode(code, static_cast<byte*>(result));
}

Expected<void> CudaAllocator::free_async(byte* pointer, cudaStream_t stream) {
  Expected<void> result = ExpectedOrCode(free_async_abi(static_cast<void*>(pointer), stream));
  if (!result) {
    return result;
  }
  return ExpectedOrCode(GxfEntityNotifyEventType(context(), eid(), GXF_EVENT_MEMORY_FREE));
}

}  // namespace gxf
}  // namespace nvidia
