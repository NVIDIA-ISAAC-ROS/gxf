/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/serialization/file.hpp"

#include <sys/stat.h>

#include <cstdio>
#include <ctime>
#include <string>

#include "gxf/std/gems/utils/time.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t File::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      allocator_, "allocator", "Allocator",
      "Memory allocator for stream buffer");
  result &= registrar->parameter(
      file_path_, "file_path", "File Path",
      "Path to file",
      std::string(""));
  result &= registrar->parameter(
      file_mode_, "file_mode", "File Mode",
      "Access mode for file (\"wb+\" by default)"
      "  \"r(b)\" Opens a (binary) file for reading"
      "  \"r(b)+\" Opens a (binary) file to update both reading and writing"
      "  \"w(b)\" Creates an empty (binary) file for writing"
      "  \"w(b)+\" Creates an empty (binary) file for both reading and writing"
      "  \"a(b)\" Appends to a (binary) file"
      "  \"a(b)+\" Opens a (binary) file for reading and appending",
      std::string("wb+"));
  result &= registrar->parameter(
      buffer_size_, "buffer_size", "Buffer Size",
      "Size of the stream buffer in bytes (2MB by default)",
      static_cast<size_t>(1 << 21));
  return ToResultCode(result);
}

gxf_result_t File::initialize() {
  return ToResultCode(
      buffer_.resize(allocator_, buffer_size_, MemoryStorageType::kSystem)
      .and_then([&]() {
        return !file_path_.get().empty() ? open() : Success;
      }));
}

gxf_result_t File::deinitialize() {
  if (file_ != nullptr) {
    auto result = close();
    if (!result) {
      return ToResultCode(result);
    }
  }
  return ToResultCode(buffer_.freeBuffer());
}

gxf_result_t File::write_abi(const void* data, size_t size, size_t* bytes_written) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  if (data == nullptr || bytes_written == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  if (file_ == nullptr) {
    GXF_LOG_ERROR("File is not open");
    return GXF_NULL_POINTER;
  }
  const size_t result = fwrite(data, sizeof(byte), size, file_);
  if (error()) {
    GXF_LOG_ERROR("Failed to write to file");
    GXF_LOG_DEBUG("Wrote %zu/%zu bytes", result, size);
    return GXF_FAILURE;
  }
  *bytes_written = result;
  return GXF_SUCCESS;
}

gxf_result_t File::read_abi(void* data, size_t size, size_t* bytes_read) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  if (data == nullptr || bytes_read == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  if (file_ == nullptr) {
    GXF_LOG_ERROR("File is not open");
    return GXF_NULL_POINTER;
  }
  const size_t result = fread(data, sizeof(byte), size, file_);
  if (error()) {
    GXF_LOG_ERROR("Failed to read from file");
    GXF_LOG_DEBUG("Read %zu/%zu bytes", result, size);
    return GXF_FAILURE;
  }
  *bytes_read = result;
  return GXF_SUCCESS;
}

Expected<void> File::open(const char* path, const char* mode) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  if (file_ != nullptr) {
    GXF_LOG_ERROR("File is already open");
    return Unexpected{GXF_FAILURE};
  }
  Expected<void> result1;
  if (path != nullptr && path != file_path_.get()) {
    result1 &= file_path_.set(path);
  }
  if (mode != nullptr && mode != file_mode_.get()) {
    result1 &= file_mode_.set(mode);
  }
  if (!result1) {
    return ForwardError(result1);
  }
  if (file_path_.get().empty()) {
    GXF_LOG_ERROR("File path is empty");
    return Unexpected{GXF_FAILURE};
  }
  file_ = fopen(file_path_.get().c_str(), file_mode_.get().c_str());
  if (file_ == nullptr) {
    GXF_LOG_ERROR("%s : %s", strerror(errno), file_path_.get().c_str());
    return Unexpected{GXF_FAILURE};
  }
  const int result2 = setvbuf(file_, reinterpret_cast<char*>(buffer_.pointer()),
                              buffer_.size() > 0 ? _IOFBF : _IONBF, buffer_.size());
  if (result2 != 0) {
    GXF_LOG_ERROR("%s : %s ", strerror(errno), file_path_.get().c_str());
    return Unexpected{GXF_FAILURE};
  }
  return Success;
}

Expected<void> File::close() {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  if (file_ == nullptr) {
    GXF_LOG_ERROR("File is already closed");
    return Unexpected{GXF_FAILURE};
  }
  const int result = fclose(file_);
  if (result != 0) {
    GXF_LOG_ERROR("%s", strerror(errno));
    return Unexpected{GXF_FAILURE};
  }
  file_ = nullptr;
  return Success;
}

void File::clear() {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  clearerr(file_);
}

bool File::eof() {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  return feof(file_) != 0;
}

bool File::error() {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  return ferror(file_) != 0;
}

Expected<void> File::flush() {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  if (file_ == nullptr) {
    GXF_LOG_ERROR("File is not open");
    return Unexpected{GXF_NULL_POINTER};
  }
  const int result = fflush(file_);
  if (result != 0) {
    GXF_LOG_ERROR("%s", strerror(errno));
    return Unexpected{GXF_FAILURE};
  }
  return Success;
}

Expected<void> File::seek(size_t offset) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  if (file_ == nullptr) {
    GXF_LOG_ERROR("File is not open");
    return Unexpected{GXF_NULL_POINTER};
  }
  const ssize_t result = fseek(file_, offset, SEEK_SET);
  if (result != 0) {
    GXF_LOG_ERROR("%s", strerror(errno));
    return Unexpected{GXF_FAILURE};
  }
  return Success;
}

Expected<size_t> File::tell() {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  if (file_ == nullptr) {
    GXF_LOG_ERROR("File is not open");
    return Unexpected{GXF_NULL_POINTER};
  }
  const ssize_t result = ftell(file_);
  if (result < 0) {
    GXF_LOG_ERROR("%s", strerror(errno));
    return Unexpected{GXF_FAILURE};
  }
  return static_cast<size_t>(result);
}

bool File::isOpen() {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  return file_ != nullptr;
}

const char* File::path() {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  return !file_path_.get().empty() ? file_path_.get().c_str() : nullptr;
}

const char* File::mode() {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  return file_mode_.get().c_str();
}

Expected<void> File::rename(const char* path) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  const int result = std::rename(file_path_.get().c_str(), path);
  if (result != 0) {
    GXF_LOG_ERROR("%s", strerror(errno));
    return Unexpected{GXF_FAILURE};
  }
  return file_path_.set(path);
}

Expected<void> File::addTimestamp(int64_t timestamp, bool utc) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  char str[64];
  std::time_t time = TimestampToTime(timestamp);
  std::strftime(str, sizeof(str), "%F_%H-%M-%S_", utc ? std::gmtime(&time) : std::localtime(&time));
  const size_t split = file_path_.get().find_last_of("/") + 1;
  const std::string path = file_path_.get().substr(0, split) + str + file_path_.get().substr(split);
  return rename(path.c_str());
}

Expected<void> File::writeProtect() {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  const int result = chmod(file_path_.get().c_str(), S_IRUSR | S_IRGRP | S_IROTH);
  if (result != 0) {
    GXF_LOG_ERROR("%s", strerror(errno));
    return Unexpected{GXF_FAILURE};
  }
  return Success;
}

}  // namespace gxf
}  // namespace nvidia
