/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/serialization/entity_replayer.hpp"

#include <string>

namespace nvidia {
namespace gxf {

gxf_result_t EntityReplayer::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      transmitter_, "transmitter", "Entity transmitter",
      "Transmitter channel for replaying entities");
  result &= registrar->parameter(
      entity_serializer_, "entity_serializer", "Entity serializer",
      "Serializer for serializing entities");
  result &= registrar->parameter(
      boolean_scheduling_term_, "boolean_scheduling_term", "BooleanSchedulingTerm",
      "BooleanSchedulingTerm to stop the codelet from ticking after all messages are published.");
  result &= registrar->parameter(
      directory_, "directory", "Directory path",
      "Directory path for storing files");
  result &= registrar->parameter(
      basename_, "basename", "Base file name",
      "User specified file name without extension",
      Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      batch_size_, "batch_size", "Batch Size",
      "Number of entities to read and publish for one tick", 1UL);
  result &= registrar->parameter(
      ignore_corrupted_entities_, "ignore_corrupted_entities", "Ignore Corrupted Entities",
      "If an entity could not be deserialized, it is ignored by default; otherwise a failure is "
      "generated.", true);
  return ToResultCode(result);
}

gxf_result_t EntityReplayer::initialize() {
  // Create path by appending component name to directory path if basename is not provided
  std::string path = directory_.get() + '/';
  if (const auto& basename = basename_.try_get()) {
    path += basename.value();
  } else {
    path += name();
  }

  // Filenames for index and data
  const std::string index_filename = path + FileStream::kIndexFileExtension;
  const std::string entity_filename = path + FileStream::kBinaryFileExtension;

  // Open index file stream as read-only
  index_file_stream_ = FileStream(index_filename, "");
  Expected<void> result = index_file_stream_.open();
  if (!result) {
    GXF_LOG_WARNING("Could not open index file: %s", index_filename.c_str());
    return ToResultCode(result);
  }

  // Open entity file stream as read-only
  entity_file_stream_ = FileStream(entity_filename, "");
  result = entity_file_stream_.open();
  if (!result) {
    GXF_LOG_WARNING("Could not open entity file: %s", entity_filename.c_str());
    return ToResultCode(result);
  }

  boolean_scheduling_term_->enable_tick();

  return GXF_SUCCESS;
}

gxf_result_t EntityReplayer::deinitialize() {
  // Close binary file stream
  Expected<void> result = entity_file_stream_.close();
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

gxf_result_t EntityReplayer::tick() {
  for (size_t i = 0; i < batch_size_; i++) {
    // Read entity index from index file
    // Break if index not found and clear stream errors
    EntityIndex index;
    Expected<size_t> size = index_file_stream_.readTrivialType(&index);
    if (!size) {
      GXF_LOG_INFO("Reach end of file. Stop ticking.");
      boolean_scheduling_term_->disable_tick();
      index_file_stream_.clear();
      break;
    }

    // Read entity from binary file
    Expected<Entity> entity = entity_serializer_->deserializeEntity(context(),
                                                                    &entity_file_stream_);
    if (!entity) {
      if (ignore_corrupted_entities_) {
        continue;
      } else {
        return ToResultCode(entity);
      }
    }

    // Publish entity
    Expected<void> result = transmitter_->publish(entity.value());
    if (!result) {
      return ToResultCode(result);
    }
  }

  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
