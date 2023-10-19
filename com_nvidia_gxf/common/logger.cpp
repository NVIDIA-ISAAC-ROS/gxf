/*
Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "logger.hpp"

#include <sys/time.h>

#include <cstdlib>
#include <ctime>

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
  "\033[1;3;31m%s.%03ld PANIC %s@%d: %s\033[0m\n",
    "\033[1;31m%s.%03ld ERROR %s@%d: %s\033[0m\n",
      "\033[33m%s.%03ld WARN  %s@%d: %s\033[0m\n",
       "\033[0m%s.%03ld INFO  %s@%d: %s\033[0m\n",
      "\033[90m%s.%03ld DEBUG %s@%d: %s\033[0m\n",
      "\033[34m%s.%03ld VERB  %s@%d: %s\033[0m\n",
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
    std::fprintf(stderr, "DefaultConsoleLogging: Invalid severity %d.", severity_int);
    std::abort();
  } else {
    return severity_int;
  }
}

}  // namespace

void (*LoggingFunction)(const char*, int, Severity, const char*, void*) = DefaultConsoleLogging;
void* LoggingFunctionArg = nullptr;

void DefaultConsoleLogging(const char* file, int line, Severity severity,
                           const char* log, void* arg) {
  if (severity == Severity::ALL || severity == Severity::COUNT) {
    std::fprintf(stderr, "DefaultConsoleLogging: Log severity cannot be 'ALL' or 'COUNT'.");
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
  if (outstream == nullptr) {
    return;
  }

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

void Redirect(std::FILE* file, Severity severity) {
  if (severity == Severity::COUNT) {
    std::fprintf(stderr, "SetSeverity: Log severity cannot be 'COUNT'.\n");
    std::abort();
  } else if (severity == Severity::NONE) {
    return;
  } else if (severity == Severity::ALL) {
    for (int i = 0; i < kNumSeverity; i++) {
      s_sinks[i] = file;
    }
  } else {
    s_sinks[SeverityToIndex(severity)] = file;
  }
}

void SetSeverity(Severity severity) {
  if (severity == Severity::COUNT) {
    std::fprintf(stderr, "SetSeverity: Log severity cannot be 'COUNT'.\n");
    std::abort();
  }

  gxf::Singleton<SeverityContainer>::Get().r = severity;
}

Severity GetSeverity() {
  return gxf::Singleton<SeverityContainer>::Get().r;
}

}  // namespace nvidia
