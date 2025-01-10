/*
Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "logger.hpp"

#include <sys/time.h>

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <string>

#include "gxf/logger/gxf_logger.hpp"

namespace nvidia {

void Redirect(std::FILE* file, Severity severity) {
  logger::GxfLogger& logger = logger::GlobalGxfLogger::instance();
  logger.redirect(static_cast<int>(severity), file);
}

bool SetSeverityFromEnv(const char* env_name) {
  return logger::GlobalGxfLogger::SetSeverityFromEnv(env_name);
}

Severity GetSeverityFromEnv(const char* env_name, int* error_code) {
  return logger::GlobalGxfLogger::GetSeverityFromEnv(env_name, error_code);
}

void SetSeverity(Severity severity) {
  logger::GxfLogger& logger = logger::GlobalGxfLogger::instance();
  logger.level(static_cast<int>(severity));
}

Severity GetSeverity() {
  logger::GxfLogger& logger = logger::GlobalGxfLogger::instance();
  int level = logger.level();
  return static_cast<Severity>(level);
}

}  // namespace nvidia
