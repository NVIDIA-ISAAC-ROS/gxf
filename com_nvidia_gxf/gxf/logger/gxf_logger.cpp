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

#include "gxf_logger.hpp"

#include <sys/time.h>

#include <algorithm>
#include <memory>
#include <string>

#include "common/singleton.hpp"

namespace nvidia {

namespace {

// File streams for writing log messages for each level of severity
std::FILE* s_sinks[] = {
    stderr,  // PANIC
    stderr,  // ERROR
    stdout,  // WARNING
    stdout,  // INFO
    stdout,  // DEBUG
    stdout,  // VERBOSE
};

// String patterns used to format the log message
// Documentation for console coloring: https://misc.flogisoft.com/bash/tip_colors_and_formatting
constexpr const char* s_patterns[] = {
    "\033[1;3;31m%s.%03ld PANIC %s@%d: %s\033[0m\n", "\033[1;31m%s.%03ld ERROR %s@%d: %s\033[0m\n",
    "\033[33m%s.%03ld WARN  %s@%d: %s\033[0m\n",     "\033[0m%s.%03ld INFO  %s@%d: %s\033[0m\n",
    "\033[90m%s.%03ld DEBUG %s@%d: %s\033[0m\n",     "\033[34m%s.%03ld VERB  %s@%d: %s\033[0m\n",
};

// Assert that there is a sink for every level of severity
constexpr int kNumSeverity = static_cast<int>(Severity::COUNT);
static_assert(kNumSeverity == sizeof(s_sinks) / sizeof(std::FILE*),
              "Need exactly one file stream per log level.");
static_assert(kNumSeverity == sizeof(s_patterns) / sizeof(char*),
              "Need exactly one fprintf pattern per log level.");

// Converts a severity level to the severity index which can be used for example to lookup into
// the arrays 's_sinks' and 's_patterns'.
int SeverityToIndex(Severity severity) {
  const int severity_int = static_cast<int>(severity);
  if (severity_int < 0 || kNumSeverity <= severity_int) {
    std::fprintf(stderr, "DefaultConsoleLogging: Invalid severity %d.\n", severity_int);
    std::abort();
  } else {
    return severity_int;
  }
}

}  // namespace

void DefaultConsoleLogging(const char* file, int line, Severity severity, const char* log,
                           void* arg) {
  if (severity == Severity::ALL || severity == Severity::COUNT) {
    std::fprintf(stderr, "DefaultConsoleLogging: Log severity cannot be 'ALL' or 'COUNT'.\n");
    std::abort();
  }

  // Ignore severity if requested
  if (Severity::ALL != gxf::Singleton<SeverityContainer>::Get().r &&
      (Severity::NONE == gxf::Singleton<SeverityContainer>::Get().r ||
       severity > gxf::Singleton<SeverityContainer>::Get().r)) {
    return;
  }

  const int severity_int = SeverityToIndex(severity);

  // Find the filestream on which to print the message
  std::FILE* outstream = s_sinks[severity_int];
  if (outstream == nullptr) { return; }

  // Create a string with current date and time
  struct tm* tm_info;
  struct timeval tv;
  gettimeofday(&tv, NULL);
  tm local_tm;
  tm_info = localtime_r(&tv.tv_sec, &local_tm);
  char time_str[20];
  strftime(time_str, 20, "%Y-%m-%d %H:%M:%S", tm_info);

  // Print the log message to the stream
  // Patterns have the following arguments: time, time in ms, file, line, message
  std::fprintf(outstream, s_patterns[severity_int], time_str, tv.tv_usec / 1000, file, line, log);
  std::fflush(outstream);
}

void (*LoggingFunction)(const char*, int, Severity, const char*, void*) = DefaultConsoleLogging;
void* LoggingFunctionArg = nullptr;

SeverityContainer::SeverityContainer() {
  int error_code = 0;
  Severity severity = logger::GlobalGxfLogger::GetSeverityFromEnv(kGxfLogEnvName, &error_code);

  if (severity != Severity::COUNT) {
    // Set the severity if the environment variable is set and valid
    r = severity;
  } else if (error_code == 1) {  // Invalid log level from the environment variable
    const char* gxf_log_env_value = std::getenv(kGxfLogEnvName);
    std::fprintf(stderr,
                 "SeverityContainer: Invalid log level '%s' from the environment variable "
                 "'%s'. Supported levels are 'VERBOSE', 'DEBUG', 'INFO', 'WARNING', "
                 "'ERROR', 'NONE'. (case insensitive)\n",
                 gxf_log_env_value, kGxfLogEnvName);
    std::abort();
  }
}

/// Namespace for the NVIDIA logger functionality.
namespace logger {

static void ensure_log_level(int level) {
  if (level < static_cast<int>(Severity::NONE) || level > static_cast<int>(Severity::COUNT)) {
    std::fprintf(stderr, "DefaultConsoleLogging: Invalid log level %d.", level);
    std::abort();
  }
}

/// Default GXF Logger implementation.
class DefaultGxfLogger : public ILogger {
 public:
  void log(const char* file, int line, const char* name, int level, const char* message,
           void* arg = nullptr) override;

  void pattern(const char* pattern) override;
  const char* pattern() const override;

  void level(int level) override;
  int level() const override;

  void redirect(int level, void* output) override;
  void* redirect(int level) const override;
};

GxfLogger::GxfLogger(const std::shared_ptr<ILogger>& logger, const LogFunction& func)
    : Logger(logger, func) {
  if (logger_ == nullptr && func_ == nullptr) { logger_ = std::make_shared<DefaultGxfLogger>(); }

  // Set level
  level(logger_->level());

  // Set pattern
  pattern(logger_->pattern());

  // Set sinks
  for (int severity = kNumSeverity - 1; severity >= 0; --severity) {
    redirect(severity, s_sinks[severity]);
  }
}

GxfLogger& GlobalGxfLogger::instance() {
  static GxfLogger instance;
  return instance;
}

bool GlobalGxfLogger::SetSeverityFromEnv(const char* env_name) {
  int error_code = 0;
  Severity severity = GetSeverityFromEnv(env_name, &error_code);

  // Print error message if the environment variable is set but invalid
  if (error_code != 0) {
    const char* gxf_log_env_value = std::getenv(env_name);
    std::fprintf(stderr,
                 "SetSeverityFromEnv: Invalid log level '%s'. Supported levels are "
                 "'VERBOSE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'NONE'. "
                 "(case insensitive)\n",
                 gxf_log_env_value);
    return false;
  }

  // Return false if the environment variable is empty or not set
  if (severity == Severity::COUNT) {
    std::fprintf(stderr, "SetSeverityFromEnv: Environment variable '%s' is empty or not set.\n",
                 env_name);
    return false;
  }

  // Set the severity if the environment variable is set and valid
  logger::GxfLogger& logger = instance();
  logger.level(static_cast<int>(severity));

  return true;
}

Severity GlobalGxfLogger::GetSeverityFromEnv(const char* env_name, int* error_code) {
  Severity severity = Severity::COUNT;

  if (error_code) { *error_code = 0; }

  const char* gxf_log_env_value = std::getenv(env_name);
  if (gxf_log_env_value && gxf_log_env_value[0] != '\0') {
    std::string log_level(gxf_log_env_value);
    std::transform(log_level.begin(), log_level.end(), log_level.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    if (log_level == "VERBOSE") {
      severity = Severity::VERBOSE;
    } else if (log_level == "DEBUG") {
      severity = Severity::DEBUG;
    } else if (log_level == "INFO") {
      severity = Severity::INFO;
    } else if (log_level == "WARNING") {
      severity = Severity::WARNING;
    } else if (log_level == "ERROR") {
      severity = Severity::ERROR;
    } else if (log_level == "NONE") {
      severity = Severity::NONE;
    } else {
      if (error_code) {
        *error_code = 1;  // Invalid log level
      }
    }
  }

  return severity;
}

void DefaultGxfLogger::log(const char* file, int line, const char* name, int level, const char* log,
                           void* arg) {
  ensure_log_level(level);

  const Severity severity = static_cast<Severity>(level);
  if (LoggingFunction) {
    LoggingFunction(file, line, severity, log, LoggingFunctionArg);
  } else {
    DefaultConsoleLogging(file, line, severity, log, arg);
  }
}

void DefaultGxfLogger::pattern(const char* pattern) {
  // Do nothing, as the default logger does not support custom patterns
}

const char* DefaultGxfLogger::pattern() const {
  // Always return the empty string, as the default logger does not support custom patterns
  return "";
}

void DefaultGxfLogger::level(int level) {
  ensure_log_level(level);
  Severity severity = static_cast<Severity>(level);

  if (severity == Severity::COUNT) {
    std::fprintf(stderr, "SetSeverity: Log severity cannot be 'COUNT'.\n");
    std::abort();
  }

  gxf::Singleton<SeverityContainer>::Get().r = severity;
}

int DefaultGxfLogger::level() const {
  int level = static_cast<int>(gxf::Singleton<SeverityContainer>::Get().r);
  return level;
}

void DefaultGxfLogger::redirect(int level, void* output) {
  ensure_log_level(level);
  const Severity severity = static_cast<Severity>(level);

  if (severity == Severity::COUNT) {
    std::fprintf(stderr, "SetSeverity: Log severity cannot be 'COUNT'.\n");
    std::abort();
  } else if (severity == Severity::NONE) {
    return;
  } else if (severity == Severity::ALL) {
    for (int i = 0; i < kNumSeverity; i++) { s_sinks[i] = reinterpret_cast<std::FILE*>(output); }
  } else {
    s_sinks[SeverityToIndex(severity)] = reinterpret_cast<std::FILE*>(output);
  }
}

void* DefaultGxfLogger::redirect(int level) const {
  ensure_log_level(level);
  const Severity severity = static_cast<Severity>(level);

  if (severity == Severity::COUNT) {
    std::fprintf(stderr, "SetSeverity: Log severity cannot be 'COUNT'.\n");
    std::abort();
  } else if (severity == Severity::NONE) {
    return nullptr;
  } else if (severity == Severity::ALL) {
    std::fprintf(stderr, "SetSeverity: Log severity cannot be 'ALL'.\n");
    std::abort();
  } else {
    return s_sinks[SeverityToIndex(severity)];
  }
}

}  // namespace logger

}  // namespace nvidia
