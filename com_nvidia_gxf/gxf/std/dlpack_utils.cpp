/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
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

#include "gxf/std/dlpack_utils.hpp"

#include <cuda_runtime.h>

#include <cinttypes>
#include <string>
#include <vector>

// #include "common/assert.hpp"
#include "common/logger.hpp"

namespace nvidia {
namespace gxf {

namespace {
// copy of CHECK_CUDA_ERROR from gxf/cuda/cuda_common.h
#define CHECK_CUDA_ERROR(cu_result, fmt, ...)                                                    \
  do {                                                                                           \
    cudaError_t err = (cu_result);                                                               \
    if (err != cudaSuccess) {                                                                    \
      GXF_LOG_ERROR(fmt ", cuda_error: %s, error_str: %s", ##__VA_ARGS__, cudaGetErrorName(err), \
                    cudaGetErrorString(err));                                                    \
      return Unexpected{GXF_FAILURE};                                                            \
    }                                                                                            \
  } while (0)

}  // namespace

Expected<DLDevice> DLDeviceFromPointer(void* ptr) {
  cudaError_t cuda_status;

  DLDevice device{.device_type = kDLCUDA, .device_id = 0};

  cudaPointerAttributes attributes;
  cuda_status = cudaPointerGetAttributes(&attributes, ptr);
  CHECK_CUDA_ERROR(cuda_status, "Unable to get pointer attributes from %p", ptr);

  switch (attributes.type) {
    case cudaMemoryTypeUnregistered:
      device.device_type = kDLCPU;
      break;
    case cudaMemoryTypeHost:
      device = {.device_type = kDLCUDAHost, .device_id = attributes.device};
      break;
    case cudaMemoryTypeDevice:
      device = {.device_type = kDLCUDA, .device_id = attributes.device};
      break;
    case cudaMemoryTypeManaged:
      device = {.device_type = kDLCUDAManaged, .device_id = attributes.device};
      break;
  }
  return device;
}

void ComputeDLPackStrides(const DLTensor& tensor, std::vector<int64_t>& strides,
                          bool to_num_elements) {
  int64_t ndim = tensor.ndim;
  strides.resize(ndim);
  int64_t elem_size = (to_num_elements) ? 1 : tensor.dtype.bits / 8;
  if (tensor.strides == nullptr) {
    int64_t step = 1;
    for (int64_t index = ndim - 1; index >= 0; --index) {
      strides[index] = step * elem_size;
      step *= tensor.shape[index];
    }
  } else {
    for (int64_t index = 0; index < ndim; ++index) {
      strides[index] = tensor.strides[index] * elem_size;
    }
  }
}

Expected<DLDataType> DLDataTypeFromTypeString(const std::string& typestr) {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes = 1;
  if (typestr.substr(0, 1) == ">") {
    GXF_LOG_ERROR("big endian types not supported");
    return Unexpected{GXF_INVALID_DATA_FORMAT};
  }
  std::string kind = typestr.substr(1, 1);
  if (kind == "i") {
    code = kDLInt;
  } else if (kind == "u") {
    code = kDLUInt;
  } else if (kind == "f") {
    code = kDLFloat;
  } else if (kind == "c") {
    code = kDLComplex;
  } else {
    GXF_LOG_ERROR("dtype.kind (%s) is not supported!", kind.c_str());
    return Unexpected{GXF_INVALID_DATA_FORMAT};
  }
  bits = std::stoi(typestr.substr(2)) * 8;
  DLDataType data_type{.code = code, .bits = bits, .lanes = lanes};
  return data_type;
}

Expected<const char*> numpyTypestr(const DLDataType dtype) {
  // future work: consider bfloat16: https://github.com/dmlc/dlpack/issues/45
  // future work: consider other byte-order?
  uint8_t code = dtype.code;
  uint8_t bits = dtype.bits;
  uint16_t lanes = dtype.lanes;
  if (lanes != 1) {
    GXF_LOG_ERROR(
      "DLDataType->NumPy typestring conversion only support DLDataType with one lane, "
      "but found dtype.lanes: (%" PRIu16 ").", lanes);
  }
  switch (code) {
    case kDLInt:
      switch (bits) {
        case 8:
          return "|i1";
        case 16:
          return "<i2";
        case 32:
          return "<i4";
        case 64:
          return "<i8";
      }
      GXF_LOG_ERROR("DLDataType(code: kDLInt, bits: (%" PRIu8 ") is not supported!", bits);
      return Unexpected{GXF_INVALID_DATA_FORMAT};
    case kDLUInt:
      switch (bits) {
        case 8:
          return "|u1";
        case 16:
          return "<u2";
        case 32:
          return "<u4";
        case 64:
          return "<u8";
      }
      GXF_LOG_ERROR("DLDataType(code: kDLUInt, bits (%" PRIu8 ") is not supported!", bits);
      return Unexpected{GXF_INVALID_DATA_FORMAT};
    case kDLFloat:
      switch (bits) {
        case 16:
          return "<f2";
        case 32:
          return "<f4";
        case 64:
          return "<f8";
      }
      GXF_LOG_ERROR("DLDataType(code: kDLFloat, bits (%" PRIu8 ") is not supported!", bits);
      return Unexpected{GXF_INVALID_DATA_FORMAT};
    case kDLComplex:
      switch (bits) {
        case 64:
          return "<c8";
        case 128:
          return "<c16";
      }
      GXF_LOG_ERROR("DLDataType(code: kDLComplex, bits (%" PRIu8 ") is not supported!", bits);
      return Unexpected{GXF_INVALID_DATA_FORMAT};
  }
  GXF_LOG_ERROR("DLDataType code (%" PRIu8 ") is not supported!", code);
  return Unexpected{GXF_INVALID_DATA_FORMAT};
}

}  // namespace gxf
}  // namespace nvidia
