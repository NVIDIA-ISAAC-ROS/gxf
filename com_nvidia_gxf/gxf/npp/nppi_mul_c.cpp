/*
Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/npp/nppi_mul_c.hpp"

#include <vector>

#include "npp.h"  // NOLINT

#include "gxf/std/tensor.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t NppiMulC::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(in_, "in");
  result &= registrar->parameter(factor_, "factor");
  result &= registrar->parameter(pool_, "pool");
  result &= registrar->parameter(out_, "out");
  return ToResultCode(result);
}

gxf_result_t NppiMulC::tick() {
  // Read message from receiver
  const auto in_message = in_->receive();
  if (!in_message) {
    return in_message.error();
  }

  // Get tensor attached to the message
  auto in_tensor = in_message.value().get<Tensor>();
  if (!in_tensor) {
    return in_tensor.error();
  }
  if (in_tensor.value()->storage_type() != MemoryStorageType::kDevice) {
    return GXF_MEMORY_INVALID_STORAGE_MODE;
  }
  auto in_tensor_ptr = in_tensor.value()->data<float>();
  if (!in_tensor_ptr) {
    return in_tensor_ptr.error();
  }
  const Shape shape = in_tensor.value()->shape();

  // Create output message
  Expected<Entity> out_message = CreateTensorMap(
      context(), pool_, {{"", MemoryStorageType::kDevice, shape, PrimitiveType::kFloat32}});
  if (!out_message) {
    return out_message.error();
  }
  auto out_tensor = out_message.value().get<Tensor>();
  if (!out_tensor) {
    return out_tensor.error();
  }
  auto out_tensor_ptr = out_tensor.value()->data<float>();
  if (!out_tensor_ptr) {
    return out_tensor_ptr.error();
  }

  // Multiple tensor with constant using NPP
  const std::vector<double> factor = factor_;
  if (static_cast<int32_t>(factor.size()) != shape.dimension(2)) {
    return GXF_ARGUMENT_INVALID;
  }
  if (shape.dimension(2) == 3) {
    const float factor[3] = {static_cast<float>(factor[0]), static_cast<float>(factor[1]),
                             static_cast<float>(factor[2])};
    const NppiSize roi = {static_cast<int>(shape.dimension(1)),
                          static_cast<int>(shape.dimension(0))};
    const int32_t row_step = 3 * sizeof(float) * shape.dimension(1);
    const NppStatus status = nppiMulC_32f_C3R(in_tensor_ptr.value(), row_step, factor,
                                              out_tensor_ptr.value(), row_step, roi);
    if (status != NPP_SUCCESS) {
      GXF_LOG_ERROR("NPP multiply operation failed");
      return GXF_FAILURE;
    }
  } else {
    return GXF_NOT_IMPLEMENTED;
  }

  // Publish output message
  return ToResultCode(out_->publish(out_message.value()));
}

}  // namespace gxf
}  // namespace nvidia
