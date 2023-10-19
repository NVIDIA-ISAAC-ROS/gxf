/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gxf/test/components/tensor_generator.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <limits>
#include <string>
#include <vector>

#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace gxf {
namespace test {

gxf_result_t TensorGenerator::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(output_, "output");
  result &= registrar->parameter(allocator_, "allocator");
  result &= registrar->parameter(shape_, "shape");
  result &= registrar->parameter(storage_type_, "storage_type", "", "",
                                 static_cast<int32_t>(MemoryStorageType::kHost));
  result &= registrar->parameter(enable_timestamps_, "enable_timestamps", "", "", true);
  result &= registrar->parameter(tensor_name_, "tensor_name", "Tensor Name",
                                 "The name of the tensor in the output message.",
                                 std::string("tensor"));
  result &= registrar->parameter(timestamp_name_, "timestamp_name", "Timestamp Name",
                                 "The name of the timestamp in the output message.",
                                 std::string("timestamp"));
  result &= registrar->parameter(number_of_tensors_, "number_of_tensors", "Number of Tensors",
                                 "number of tensors to be generated in the message.",
                                 1);
  return ToResultCode(result);
}

gxf_result_t TensorGenerator::tick() {
  auto message = Entity::New(context());
  if (!message) {
    return ToResultCode(message);
  }
  const uint64_t acqtime = std::chrono::system_clock::now().time_since_epoch().count();
  for (int i = 0; i < number_of_tensors_; i++) {
    auto tensor = message.value().add<Tensor>(tensor_name_.get().c_str());
    if (!tensor) {
      return ToResultCode(tensor);
    }

    uint32_t rank = shape_.get().size();
    if (rank > Shape::kMaxRank) {
      return GXF_FAILURE;
    }
    std::array<int32_t, Shape::kMaxRank> dims;
    std::copy(std::begin(shape_.get()), std::end(shape_.get()), std::begin(dims));

    auto result = tensor.value()->reshape<DataType>(Shape(dims, rank),
                                                    MemoryStorageType(storage_type_.get()),
                                                    allocator_);
    if (!result) {
      return ToResultCode(result);
    }

    auto min = std::numeric_limits<DataType>::min();
    auto max = std::numeric_limits<DataType>::max();
    std::uniform_real_distribution<DataType> distribution(min, max);
    std::vector<DataType> elements;
    for (size_t idx = 0; idx < tensor.value()->element_count(); idx++) {
      elements.push_back(distribution(generator_));
    }

    const cudaMemcpyKind operation = (tensor.value()->storage_type() == MemoryStorageType::kHost ||
                                      tensor.value()->storage_type() == MemoryStorageType::kSystem)
      ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice;
    const cudaError_t error = cudaMemcpy(tensor.value()->pointer(), elements.data(),
                                        tensor.value()->size(), operation);
    if (error != cudaSuccess) {
      GXF_LOG_ERROR("Failure in cudaMemcpy. cuda_error: %s, error_str: %s",
                    cudaGetErrorName(error), cudaGetErrorString(error));
      return GXF_FAILURE;
    }
  }
  const uint64_t pubtime = std::chrono::system_clock::now().time_since_epoch().count();
  if (enable_timestamps_) {
    auto timestamp = message.value().add<Timestamp>(timestamp_name_.get().c_str());
    if (!timestamp) {
      return ToResultCode(timestamp);
    }
    timestamp.value()->acqtime = acqtime;
    timestamp.value()->pubtime = pubtime;
  }
  auto result = output_->publish(message.value());
  if (!result) {
    return ToResultCode(result);
  }

  return GXF_SUCCESS;
}

}  // namespace test
}  // namespace gxf
}  // namespace nvidia
