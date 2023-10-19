/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/file_stream_logger.hpp"

#include <cstdio>

namespace nvidia {
namespace gxf {

FileStreamLogger::FileStreamLogger() : handle_(nullptr) {}

gxf_result_t FileStreamLogger::open(const char* filename) {
  handle_ = std::fopen(filename, "w");
  if (handle_ == nullptr) {
    std::fprintf(
        stderr, "\033[1;31m[WARNING] Couldn't open the file:%s. Using /dev/null for logs.\033[0m\n",
        filename);
    handle_ = std::fopen("/dev/null", "w");
  }
  return GXF_SUCCESS;
}

void FileStreamLogger::operator()(const char* fmt) {
  if (handle_ == nullptr) {
    return;
  }
  std::fprintf(handle_, "%s", fmt);
  std::fprintf(handle_, "\n");
  std::fflush(handle_);
}

}  // namespace gxf
}  // namespace nvidia
