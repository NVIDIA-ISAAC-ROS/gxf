/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/serialization/file_stream.hpp"

namespace nvidia {
namespace gxf {

Expected<void> FileStream::open() {
  if (!output_file_path_.empty()) {
    output_file_.open(output_file_path_.c_str(), std::ios::out | std::ios::binary);
  }
  if (!input_file_path_.empty()) {
    input_file_.open(input_file_path_.c_str(), std::ios::in | std::ios::binary);
  }
  return input_file_ && output_file_ ? Success : Unexpected{GXF_FAILURE};
}

Expected<void> FileStream::close() {
  if (input_file_.is_open()) {
    input_file_.close();
  }
  if (output_file_.is_open()) {
    output_file_.close();
  }
  return input_file_ && output_file_ ? Success : Unexpected{GXF_FAILURE};
}

gxf_result_t FileStream::write_abi(const void* data, size_t size, size_t* bytes_written) {
  if (data == nullptr || bytes_written == nullptr) { return GXF_ARGUMENT_NULL; }
  output_file_.write(static_cast<const char*>(data), size);
  *bytes_written = size;
  return output_file_ ? GXF_SUCCESS : GXF_FAILURE;
}

gxf_result_t FileStream::read_abi(void* data, size_t size, size_t* bytes_read) {
  if (data == nullptr || bytes_read == nullptr) { return GXF_ARGUMENT_NULL; }
  input_file_.read(static_cast<char*>(data), size);
  *bytes_read = size;
  return input_file_ ? GXF_SUCCESS : GXF_FAILURE;
}

void FileStream::clear() {
  input_file_.clear();
  output_file_.clear();
}

Expected<void> FileStream::flush() {
  output_file_.flush();
  if (!output_file_) {
    return Unexpected{GXF_FAILURE};
  }
  return Success;
}

Expected<void> FileStream::setWriteOffset(size_t index) {
  output_file_.seekp(index);
  if (!output_file_) {
    return Unexpected{GXF_FAILURE};
  }
  return Success;
}

Expected<size_t> FileStream::getWriteOffset() {
  size_t index = output_file_.tellp();
  if (!output_file_) {
    return Unexpected{GXF_FAILURE};
  }
  return index;
}

Expected<void> FileStream::setReadOffset(size_t index) {
  input_file_.seekg(index);
  if (!input_file_) {
    return Unexpected{GXF_FAILURE};
  }
  return Success;
}

Expected<size_t> FileStream::getReadOffset() {
  size_t index = input_file_.tellg();
  if (!input_file_) {
    return Unexpected{GXF_FAILURE};
  }
  return index;
}

}  // namespace gxf
}  // namespace nvidia
