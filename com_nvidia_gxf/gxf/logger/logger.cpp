/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gxf/logger/logger.hpp"

#include <memory>

namespace nvidia {

/// Namespace for the NVIDIA logger functionality.
namespace logger {

constexpr int kMaxSeverity = 255;

Logger::Logger(const std::shared_ptr<ILogger>& logger) : logger_(logger) {}

Logger::Logger(const LogFunction& func) : func_(func) {}

Logger::Logger(const std::shared_ptr<ILogger> logger, const LogFunction& func)
    : logger_(logger), func_(func) {}

void Logger::log(const char* file, int line, const char* name, int level, const char* message) {
  // func_ (LogFunction) has higher priority than logger_ (ILogger)
  if (func_) {
    func_(file, line, name, level, message, func_arg_);
  } else if (logger_) {
    logger_->log(file, line, name, level, message);
  }
}

void Logger::logger(std::shared_ptr<ILogger> logger) {
  logger_ = logger;
}

std::shared_ptr<ILogger> Logger::logger() const {
  return logger_;
}

void Logger::func(LogFunction log_func, void* arg) {
  func_ = log_func;
  func_arg_ = arg;
}

LogFunction Logger::func() const {
  return func_;
}

void* Logger::arg() const {
  return func_arg_;
}

void Logger::pattern(const char* pattern) {
  if (logger_) { logger_->pattern(pattern); }
  pattern_ = pattern;
}

const char* Logger::pattern() const {
  if (logger_) { return logger_->pattern(); }
  return pattern_.c_str();
}

void Logger::level(int level) {
  if (logger_) { logger_->level(level); }
  level_ = level;
}

int Logger::level() const {
  if (logger_) { return logger_->level(); }
  return level_;
}

void Logger::redirect(int level, void* output) {
  bool valid = false;
  if (level >= 0 && level <= kMaxSeverity) {
    // Resize the sinks_ vector if necessary
    if (static_cast<int>(sinks_.size()) <= level) { sinks_.resize(level + 1); }
    valid = true;
  }
  if (logger_) { logger_->redirect(level, output); }
  if (valid) { sinks_[level] = output; }
}

void* Logger::redirect(int level) const {
  if (logger_) { return logger_->redirect(level); }
  if (level >= 0 && level < static_cast<int>(sinks_.size())) { return sinks_[level]; }
  return nullptr;
}

}  // namespace logger

}  // namespace nvidia
