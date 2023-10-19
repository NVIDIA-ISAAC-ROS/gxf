/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/serialization/endpoint.hpp"

namespace nvidia {
namespace gxf {

bool Endpoint::isWriteAvailable() {
  return is_write_available_abi() == GXF_SUCCESS;
}
bool Endpoint::isReadAvailable() {
  return is_read_available_abi() == GXF_SUCCESS;
}

Expected<size_t> Endpoint::write(const void* data, size_t size) {
  size_t result;
  return ExpectedOrCode(write_abi(data, size, &result), result);
}

Expected<size_t> Endpoint::read(void* data, size_t size) {
  size_t result;
  return ExpectedOrCode(read_abi(data, size, &result), result);
}

Expected<void> Endpoint::write_ptr(const void* data, size_t size, MemoryStorageType type) {
  return ExpectedOrCode(write_ptr_abi(data, size, type));
}

}  // namespace gxf
}  // namespace nvidia
