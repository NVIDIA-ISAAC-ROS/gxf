/*
Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/npp/nppi_set.hpp"

#include <vector>

#include "npp.h"  // NOLINT

#include "gxf/std/tensor.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t NppiSet::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(rows_, "rows");
  result &= registrar->parameter(columns_, "columns");
  result &= registrar->parameter(channels_, "channels");
  result &= registrar->parameter(pool_, "pool");
  result &= registrar->parameter(value_, "value");
  result &= registrar->parameter(out_, "out");
  return ToResultCode(result);
}

gxf_result_t NppiSet::tick() {
  const int32_t rows = rows_;
  const int32_t columns = columns_;
  const int32_t channels = channels_;

  // Create output message
  Expected<Entity> out_message =
      CreateTensorMap(context(), pool_,
                      {{"", MemoryStorageType::kDevice, Shape{{rows, columns, channels}},
                        PrimitiveType::kFloat32}});
  if (!out_message) return out_message.error();
  auto out_tensor = out_message.value().get<Tensor>();
  if (!out_tensor) {
    return out_tensor.error();
  }
  auto out_tensor_ptr = out_tensor.value()->data<float>();
  if (!out_tensor_ptr) {
    return out_tensor_ptr.error();
  }

  // Set tensor to constant using NPP
  const std::vector<double> value = value_;
  if (static_cast<int32_t>(value.size()) != channels) {
    return GXF_ARGUMENT_INVALID;
  }
  if (channels == 3) {
    const float constant[3] = {static_cast<float>(value[0]), static_cast<float>(value[1]),
                               static_cast<float>(value[2])};
    const NppiSize roi = {static_cast<int>(columns), static_cast<int>(rows)};
    const int32_t row_step = 3 * sizeof(float) * columns;
    const NppStatus status = nppiSet_32f_C3R(constant, out_tensor_ptr.value(), row_step, roi);
    if (status != NPP_SUCCESS) {
      GXF_LOG_ERROR("NPP set operation failed");
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
