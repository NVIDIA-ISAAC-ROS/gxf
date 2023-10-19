/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gxf/test/components/camera_message_generator.hpp"

#include <cmath>
#include <cstring>
#include "common/byte.hpp"
#include "common/logger.hpp"
#include "cuda_runtime.h"

#include "gxf/core/gxf.h"
#include "gxf/messages/camera_message.hpp"
#include "gxf/multimedia/camera.hpp"

namespace nvidia {
namespace gxf {
namespace test {

namespace {

constexpr size_t kWidth = 3840;
constexpr size_t kHeight = 2160;
constexpr size_t kChannels = 3;

constexpr gxf::CameraModel kIntrinsics = {
  { kWidth, kHeight },
  { kWidth / 1.5, kHeight / 1.2 },
  { kWidth / 2.0, kHeight / 2.0 },
  0.0,
  gxf::DistortionType::Polynomial,
  { 0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8 },
  { 0.70259051, -0.71156708, 0.00623408,
   -0.22535139, -0.23080127, -0.94654505,
    0.67496913,  0.66362871, -0.32251142 },
  { 122.43413524, -58.4445669 , -8.71785439, 1637.28675475,
    4.54429487, 3.30940264, -134.40907701, 2880.869899,
    0.02429085, 0.02388273, -0.01160657, 1. },
};

constexpr size_t kCameraUID = 100;
constexpr size_t kSequenceNumber = 50;
constexpr size_t kTimestampAcqtime = 1000;
constexpr size_t kTimestampPubtime = 2000;
constexpr size_t kRMatrixSize = 9;
constexpr size_t kPMatrixSize = 12;

}  // namespace

// A very small floating point number close to what can be represented by the corresponding
// floating point standard.
// For more details see https://en.wikipedia.org/wiki/Machine_epsilon
template<typename K>
constexpr K MachineEpsilon = K(0);
template<>
constexpr float MachineEpsilon<float> = 5.9604644775390625e-8f;
template<>
constexpr double MachineEpsilon<double> = 1.1102230246251565404236316680908203125e-16;

// Returns true if two floating point values are so close that they can be considered to be equal
// under floating point rounding errors. This function uses relative comparison technique.
// Warning: Do not use to compare against small floats, or zero, use IsAlmostZero instead.
template<typename K>
bool IsAlmostEqualRelative(K x, K y, K max_rel_diff = K(10)*MachineEpsilon<K>) {
  const K diff = std::abs(x - y);
  const K absx = std::abs(x);
  const K absy = std::abs(y);
  const K larger = (absx > absy) ? absx : absy;
  return (diff <= larger * max_rel_diff);
}

gxf_result_t TestCameraMessage::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(camera_message_output_, "camera_message_output");
  result &= registrar->parameter(camera_message_input_, "camera_message_input");
  result &= registrar->parameter(allocator_, "allocator");
  result &= registrar->parameter(storage_type_, "storage_type");
  return gxf::ToResultCode(result);
}

gxf_result_t TestCameraMessage::initialize() {
  auto result = allocator_->allocate(kWidth * kHeight * kChannels,
                                     gxf::MemoryStorageType::kSystem)
      .assign_to(frame_);
  if (!result) {
    return gxf::ToResultCode(result);
  }
  for (size_t row = 0; row < kHeight; row++) {
    for (size_t col = 0; col < kWidth; col++) {
      for (size_t channel = 0; channel < kChannels; channel++) {
        const size_t index = (row * kWidth * kChannels) + (col * kChannels) + channel;
        frame_[index] = index % 255;
      }
    }
  }
  return GXF_SUCCESS;
}

gxf_result_t TestCameraMessage::deinitialize() {
  return gxf::ToResultCode(allocator_->free(frame_));
}

gxf_result_t TestCameraMessage::start() {
  return gxf::ToResultCode(
    CreateCameraMessage<gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB>(
        context(), kWidth, kHeight, gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR,
        static_cast<gxf::MemoryStorageType>(storage_type_.get()), allocator_)
    .map([&](CameraMessageParts message) -> gxf::Expected<void> {
      cudaMemcpyKind operation;
      switch (message.frame->storage_type()) {
        case gxf::MemoryStorageType::kSystem:
        case gxf::MemoryStorageType::kHost: {
          operation = cudaMemcpyKind::cudaMemcpyHostToHost;
        } break;
        case gxf::MemoryStorageType::kDevice: {
          operation = cudaMemcpyKind::cudaMemcpyHostToDevice;
        } break;
        default:
          return gxf::Unexpected{GXF_MEMORY_INVALID_STORAGE_MODE};
      }
      const cudaError_t error = cudaMemcpy(
          message.frame->pointer(), frame_, message.frame->size(), operation);
      if (error != cudaSuccess) {
        GXF_LOG_ERROR("%s", cudaGetErrorString(error));
        return gxf::Unexpected{GXF_FAILURE};
      }
      *message.intrinsics = kIntrinsics;
      *message.sequence_number = kSequenceNumber;
      message.timestamp->acqtime = kTimestampAcqtime;
      message.timestamp->pubtime = kTimestampPubtime;
      return camera_message_output_->publish(message.entity);
    }));
}

gxf_result_t TestCameraMessage::stop() {
  return gxf::ToResultCode(
    camera_message_input_->receive()
    .map(GetCameraMessage)
    .map([&](CameraMessageParts message) -> gxf::Expected<void> {
      gxf::Expected<void> result;
      auto buffer = allocator_->allocate(message.frame->size(),
                                         gxf::MemoryStorageType::kSystem);
      if (!buffer) {
        return gxf::ForwardError(buffer);
      }
      cudaMemcpyKind operation;
      switch (message.frame->storage_type()) {
        case gxf::MemoryStorageType::kSystem:
        case gxf::MemoryStorageType::kHost: {
          operation = cudaMemcpyKind::cudaMemcpyHostToHost;
        } break;
        case gxf::MemoryStorageType::kDevice: {
          operation = cudaMemcpyKind::cudaMemcpyDeviceToHost;
        } break;
        default:
          return gxf::Unexpected{GXF_MEMORY_INVALID_STORAGE_MODE};
      }
      const cudaError_t error = cudaMemcpy(
          buffer.value(), message.frame->pointer(), message.frame->size(), operation);
      if (error != cudaSuccess) {
        GXF_LOG_ERROR("%s", cudaGetErrorString(error));
        return gxf::Unexpected{GXF_FAILURE};
      }
      const size_t size = kWidth * kHeight * kChannels;
      if (message.frame->size() != size) {
        GXF_LOG_ERROR("Expected frame size to be %zu but got %zu", size, message.frame->size());
        result &= gxf::Expected<void>(gxf::Unexpected{GXF_FAILURE});
      }
      if (std::memcmp(buffer.value(), frame_, message.frame->size()) != 0) {
        GXF_LOG_ERROR("Frames do not match");
        result &= gxf::Expected<void>(gxf::Unexpected{GXF_FAILURE});
      }
      if (message.intrinsics->dimensions.x != static_cast<size_t>(kIntrinsics.dimensions.x)) {
        GXF_LOG_ERROR("Expected dimension x to be %u but got %u",
                      kIntrinsics.dimensions.x, message.intrinsics->dimensions.x);
        result &= gxf::Expected<void>(gxf::Unexpected{GXF_FAILURE});
      }
      if (message.intrinsics->dimensions.y != static_cast<size_t>(kIntrinsics.dimensions.y)) {
        GXF_LOG_ERROR("Expected dimension y to be %u but got %u",
                      kIntrinsics.dimensions.y, message.intrinsics->dimensions.y);
        result &= gxf::Expected<void>(gxf::Unexpected{GXF_FAILURE});
      }
      if (!IsAlmostEqualRelative<float>(
          message.intrinsics->focal_length.x, kIntrinsics.focal_length.x)) {
        GXF_LOG_ERROR("Expected focal x to be %f but got %f",
                      kIntrinsics.focal_length.x, message.intrinsics->focal_length.x);
        result &= gxf::Expected<void>(gxf::Unexpected{GXF_FAILURE});
      }
      if (!IsAlmostEqualRelative<float>(
          message.intrinsics->focal_length.y, kIntrinsics.focal_length.y)) {
        GXF_LOG_ERROR("Expected focal y to be %f but got %f",
                      kIntrinsics.focal_length.y, message.intrinsics->focal_length.y);
        result &= gxf::Expected<void>(gxf::Unexpected{GXF_FAILURE});
      }
      if (!IsAlmostEqualRelative<float>(
          message.intrinsics->principal_point.x, kIntrinsics.principal_point.x)) {
        GXF_LOG_ERROR("Expected principal point x to be %f but got %f",
                      kIntrinsics.principal_point.x, message.intrinsics->principal_point.x);
        result &= gxf::Expected<void>(gxf::Unexpected{GXF_FAILURE});
      }
      if (!IsAlmostEqualRelative<float>(
          message.intrinsics->principal_point.y, kIntrinsics.principal_point.y)) {
        GXF_LOG_ERROR("Expected principal point y to be %f but got %f",
                      kIntrinsics.principal_point.y, message.intrinsics->principal_point.y);
        result &= gxf::Expected<void>(gxf::Unexpected{GXF_FAILURE});
      }
      if (message.intrinsics->distortion_type != gxf::DistortionType::Polynomial) {
        GXF_LOG_ERROR("Expected distortion type to be %zu but got %zu",
                      static_cast<size_t>(gxf::DistortionType::Polynomial),
                      static_cast<size_t>(message.intrinsics->distortion_type));
        result &= gxf::Expected<void>(gxf::Unexpected{GXF_FAILURE});
      }
      for (size_t i = 0; i < gxf::CameraModel::kMaxDistortionCoefficients; i++) {
        if (!IsAlmostEqualRelative<float>(
            message.intrinsics->distortion_coefficients[i],
            kIntrinsics.distortion_coefficients[i])) {
          GXF_LOG_ERROR("Expected distortion coefficient %zu to be %f but got %f",
                        i, kIntrinsics.distortion_coefficients[i],
                        message.intrinsics->distortion_coefficients[i]);
          result &= gxf::Expected<void>(gxf::Unexpected{GXF_FAILURE});
        }
      }
      for (size_t i = 0; i < kRMatrixSize; i++) {
        if (!IsAlmostEqualRelative<float>(
            message.intrinsics->rectification[i],
            kIntrinsics.rectification[i])) {
          GXF_LOG_ERROR("Expected rectification coefficient %zu to be %f but got %f",
                        i, kIntrinsics.rectification[i],
                        message.intrinsics->rectification[i]);
          result &= gxf::Expected<void>(gxf::Unexpected{GXF_FAILURE});
        }
      }
      for (size_t i = 0; i < kPMatrixSize; i++) {
        if (!IsAlmostEqualRelative<float>(
            message.intrinsics->projection_camera_rect[i],
            kIntrinsics.projection_camera_rect[i])) {
          GXF_LOG_ERROR("Expected projection camera coefficient %zu to be %f but got %f",
                        i, kIntrinsics.projection_camera_rect[i],
                        message.intrinsics->projection_camera_rect[i]);
          result &= gxf::Expected<void>(gxf::Unexpected{GXF_FAILURE});
        }
      }
      if (*message.sequence_number != kSequenceNumber) {
        GXF_LOG_ERROR("Expected frame number to be %zu but got %zu",
                      kSequenceNumber, *message.sequence_number);
        result &= gxf::Expected<void>(gxf::Unexpected{GXF_FAILURE});
      }
      if (message.timestamp->acqtime != kTimestampAcqtime) {
        GXF_LOG_ERROR("Expected timestamp acqtime to be %zu but got %zu",
                      kTimestampAcqtime, message.timestamp->acqtime);
        result &= gxf::Expected<void>(gxf::Unexpected{GXF_FAILURE});
      }
      if (message.timestamp->pubtime != kTimestampPubtime) {
        GXF_LOG_ERROR("Expected timestamp pubtime to be %zu but got %zu",
                      kTimestampPubtime, message.timestamp->pubtime);
        result &= gxf::Expected<void>(gxf::Unexpected{GXF_FAILURE});
      }
      return result & allocator_->free(buffer.value());
    }));
}

}  // namespace test
}  // namespace gxf
}  // namespace nvidia
