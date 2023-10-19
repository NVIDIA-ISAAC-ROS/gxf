/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/serialization/entity_recorder.hpp"

#include <chrono>
#include <string>

namespace nvidia {
namespace gxf {

gxf_result_t EntityRecorder::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      receiver_, "receiver", "Entity receiver",
      "Receiver channel to log");
  result &= registrar->parameter(
      entity_serializer_, "entity_serializer", "Entity serializer",
      "Serializer for serializing entities");
  result &= registrar->parameter(
      directory_, "directory", "Directory path",
      "Directory path for storing files");
  result &= registrar->parameter(
      basename_, "basename", "Base file name",
      "User specified file name without extension",
      Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      flush_on_tick_, "flush_on_tick", "Flush on tick",
      "Flushes output buffer on every tick when true", false);
  return ToResultCode(result);
}

gxf_result_t EntityRecorder::initialize() {
  // Create path by appending receiver name to directory path if basename is not provided
  std::string path = directory_.get() + '/';
  if (const auto& basename = basename_.try_get()) {
    path += basename.value();
  } else {
    path += receiver_->name();
  }

  // Initialize index file stream as write-only
  index_file_stream_ = FileStream("", path + FileStream::kIndexFileExtension);

  // Initialize binary file stream as write-only
  binary_file_stream_ = FileStream("", path + FileStream::kBinaryFileExtension);

  // Open index file stream
  Expected<void> result = index_file_stream_.open();
  if (!result) {
    return ToResultCode(result);
  }

  // Open binary file stream
  result = binary_file_stream_.open();
  if (!result) {
    return ToResultCode(result);
  }
  binary_file_offset_ = 0;

  return GXF_SUCCESS;
}

gxf_result_t EntityRecorder::deinitialize() {
  // Close binary file stream
  Expected<void> result = binary_file_stream_.close();
  if (!result) {
    return ToResultCode(result);
  }

  // Close index file stream
  result = index_file_stream_.close();
  if (!result) {
    return ToResultCode(result);
  }

  return GXF_SUCCESS;
}

gxf_result_t EntityRecorder::tick() {
  // Receive entity
  Expected<Entity> entity = receiver_->receive();
  if (!entity) {
    return ToResultCode(entity);
  }

  // Write entity to binary file
  Expected<size_t> size = entity_serializer_->serializeEntity(entity.value(), &binary_file_stream_);
  if (!size) {
    return ToResultCode(size);
  }

  // Create entity index
  EntityIndex index;
  index.log_time = std::chrono::system_clock::now().time_since_epoch().count();
  index.data_size = size.value();
  index.data_offset = binary_file_offset_;

  // Write entity index to index file
  Expected<size_t> result = index_file_stream_.writeTrivialType(&index);
  if (!result) {
    return ToResultCode(result);
  }
  binary_file_offset_ += size.value();

  if (flush_on_tick_) {
    // Flush binary file output stream
    Expected<void> result = binary_file_stream_.flush();
    if (!result) {
      return ToResultCode(result);
    }

    // Flush index file output stream
    result = index_file_stream_.flush();
    if (!result) {
      return ToResultCode(result);
    }
  }

  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
