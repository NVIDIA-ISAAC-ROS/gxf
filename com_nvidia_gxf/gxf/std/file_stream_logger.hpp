/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_STD_FILE_STREAM_LOGGER_HPP
#define NVIDIA_GXF_STD_FILE_STREAM_LOGGER_HPP

#include <cstdio>

#include "gxf/core/gxf.h"

namespace nvidia {
namespace gxf {

class FileStreamLogger {
 public:
  FileStreamLogger();

  gxf_result_t open(const char* filename);

  void operator()(const char* fmt);

  template <typename... Args>
  void operator()(const char* fmt, Args&&... args) {
    if (handle_ == nullptr) {
      return;
    }
    std::fprintf(handle_, fmt, args...);
    std::fprintf(handle_, "\n");
    std::fflush(handle_);
  }

 private:
  std::FILE* handle_;
};

}  // namespace gxf
}  // namespace nvidia

#endif
