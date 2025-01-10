/*
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <cstdint>
#include <cstdlib>
#include "gtest/gtest.h"

#include "common/logger.hpp"

namespace nvidia {
namespace test {

namespace {

constexpr size_t kBufferSize = 64;

}  // namespace

TEST(TestLogger, Severity) {
  SetSeverity(Severity::NONE);
  EXPECT_EQ(GetSeverity(), Severity::NONE);

  SetSeverity(Severity::ALL);
  EXPECT_EQ(GetSeverity(), Severity::ALL);

  SetSeverity(Severity::PANIC);
  EXPECT_EQ(GetSeverity(), Severity::PANIC);

  SetSeverity(Severity::ERROR);
  EXPECT_EQ(GetSeverity(), Severity::ERROR);

  SetSeverity(Severity::WARNING);
  EXPECT_EQ(GetSeverity(), Severity::WARNING);

  SetSeverity(Severity::INFO);
  EXPECT_EQ(GetSeverity(), Severity::INFO);

  SetSeverity(Severity::DEBUG);
  EXPECT_EQ(GetSeverity(), Severity::DEBUG);

  SetSeverity(Severity::VERBOSE);
  EXPECT_EQ(GetSeverity(), Severity::VERBOSE);
}

TEST(TestLogger, DefaultLogging) {
  SetSeverity(Severity::ALL);

  GXF_LOG_PANIC("PANIC");

  GXF_LOG_ERROR("ERROR");

  GXF_LOG_WARNING("WARNING");

  GXF_LOG_INFO("INFO");

  GXF_LOG_DEBUG("DEBUG");

  GXF_LOG_VERBOSE("VERBOSE");
}

TEST(TestLogger, CustomLogging) {
  SetSeverity(Severity::ALL);

  char buffer[kBufferSize];
  LoggingFunction = [](const char* file, int line, Severity severity, const char* log, void* arg) {
    char* buffer = reinterpret_cast<char*>(arg);
    std::snprintf(buffer, kBufferSize, "%s", log);
  };
  LoggingFunctionArg = reinterpret_cast<void*>(&buffer);

  GXF_LOG_PANIC("PANIC");
  EXPECT_EQ(std::string(buffer), "PANIC");

  GXF_LOG_ERROR("ERROR");
  EXPECT_EQ(std::string(buffer), "ERROR");

  GXF_LOG_WARNING("WARNING");
  EXPECT_EQ(std::string(buffer), "WARNING");

  GXF_LOG_INFO("INFO");
  EXPECT_EQ(std::string(buffer), "INFO");

  GXF_LOG_DEBUG("DEBUG");
  EXPECT_EQ(std::string(buffer), "DEBUG");

  GXF_LOG_VERBOSE("VERBOSE");
  EXPECT_EQ(std::string(buffer), "VERBOSE");
}

TEST(TestLogger, MaxValues) {
  SetSeverity(Severity::ALL);

  char buffer[kBufferSize];
  LoggingFunction = [](const char* file, int line, Severity severity, const char* log, void* arg) {
    char* buffer = reinterpret_cast<char*>(arg);
    std::snprintf(buffer, kBufferSize, "%s", log);
  };
  LoggingFunctionArg = reinterpret_cast<void*>(&buffer);

  GXF_LOG_INFO("%ld", INT64_MAX);
  EXPECT_EQ(std::string(buffer), "9223372036854775807");

  GXF_LOG_INFO("%d", INT32_MAX);
  EXPECT_EQ(std::string(buffer), "2147483647");

  GXF_LOG_INFO("%lu", UINT64_MAX);
  EXPECT_EQ(std::string(buffer), "18446744073709551615");

  GXF_LOG_INFO("%u", UINT32_MAX);
  EXPECT_EQ(std::string(buffer), "4294967295");

  GXF_LOG_INFO("%f", FLT_MAX);
  EXPECT_EQ(std::string(buffer), "340282346638528859811704183484516925440.000000");

  GXF_LOG_INFO("%f", DBL_MAX);
  EXPECT_EQ(std::string(buffer), "179769313486231570814527423731704356798070567525844996598917476");
}

TEST(TestLogger, MinValues) {
  SetSeverity(Severity::ALL);

  char buffer[kBufferSize];
  LoggingFunction = [](const char* file, int line, Severity severity, const char* log, void* arg) {
    char* buffer = reinterpret_cast<char*>(arg);
    std::snprintf(buffer, kBufferSize, "%s", log);
  };
  LoggingFunctionArg = reinterpret_cast<void*>(&buffer);

  GXF_LOG_INFO("%ld", INT64_MIN);
  EXPECT_EQ(std::string(buffer), "-9223372036854775808");

  GXF_LOG_INFO("%d", INT32_MIN);
  EXPECT_EQ(std::string(buffer), "-2147483648");

  GXF_LOG_INFO("%lu", static_cast<uint64_t>(0));
  EXPECT_EQ(std::string(buffer), "0");

  GXF_LOG_INFO("%u", static_cast<uint32_t>(0));
  EXPECT_EQ(std::string(buffer), "0");

  GXF_LOG_INFO("%f", FLT_MIN);
  EXPECT_EQ(std::string(buffer), "0.000000");

  GXF_LOG_INFO("%f", DBL_MIN);
  EXPECT_EQ(std::string(buffer), "0.000000");
}

class EnvVarWrapper {
 public:
  /// Constructor takes a vector of pairs (name, value) of environment variables to set
  explicit EnvVarWrapper(
      const std::vector<std::pair<std::string, std::string>>& env_var_settings = {})
      : env_var_settings_(env_var_settings) {
    // Save existing environment variables and apply new ones
    for (const auto& [env_var, value] : env_var_settings_) {
      const char* orig_value = getenv(env_var.c_str());
      if (orig_value) { orig_env_vars_[env_var] = orig_value; }
      setenv(env_var.c_str(), value.c_str(), 1);
    }
  }

  /// Constructor takes a single environment variable to set
  EnvVarWrapper(std::string key, std::string value) : EnvVarWrapper({{key, value}}) {}

  /// Destructor
  ~EnvVarWrapper() {
    // Restore original environment variables
    for (const auto& [env_var, _] : env_var_settings_) {
      auto it = orig_env_vars_.find(env_var);
      if (it == orig_env_vars_.end()) {
        unsetenv(env_var.c_str());
      } else {
        setenv(env_var.c_str(), it->second.c_str(), 1);
      }
    }
  }

 private:
  std::vector<std::pair<std::string, std::string>> env_var_settings_;
  std::unordered_map<std::string, std::string> orig_env_vars_;
};

TEST(TestLogger, TestSetSeverityFromEnv) {
  // Restore the log severity to Info after the test.
  struct SetSeverityCleanup {
    ~SetSeverityCleanup() { SetSeverity(Severity::INFO); }
  } set_severity_cleanup;

  // Set the default logging function.
  LoggingFunction = DefaultConsoleLogging;

  const std::pair<std::string, bool> log_level_list[] = {
      {"PANIC", false},  {"ERROR", true}, {"WARNING", true}, {"INFO", true}, {"DEBUG", true},
      {"VERBOSE", true}, {"NONE", true},  {"ALL", false},    {"", false},
  };

  // Test all possible log levels with log_level_list
  for (size_t i = 0; i < sizeof(log_level_list) / sizeof(log_level_list[0]); i++) {
    // Set the environment variable temporarily.
    // It will be restored when the wrapper goes out of scope.
    EnvVarWrapper wrapper(kGxfLogEnvName, log_level_list[i].first.c_str());
    // Set severity from environment variable.
    bool result = SetSeverityFromEnv();

    // Check if the severity is set correctly.
    if (log_level_list[i].second) {
      EXPECT_TRUE(result) << "log_level_list[" << i << "].first = " << log_level_list[i].first;
    } else {
      EXPECT_FALSE(result) << "log_level_list[" << i << "].first = " << log_level_list[i].first;
    }
    if (result) {
      if (log_level_list[i].first == "PANIC") {
        EXPECT_EQ(GetSeverity(), Severity::PANIC)
            << "log_level_list[" << i << "].first = " << log_level_list[i].first;
      } else if (log_level_list[i].first == "ERROR") {
        EXPECT_EQ(GetSeverity(), Severity::ERROR)
            << "log_level_list[" << i << "].first = " << log_level_list[i].first;
      } else if (log_level_list[i].first == "WARNING") {
        EXPECT_EQ(GetSeverity(), Severity::WARNING)
            << "log_level_list[" << i << "].first = " << log_level_list[i].first;
      } else if (log_level_list[i].first == "INFO") {
        EXPECT_EQ(GetSeverity(), Severity::INFO)
            << "log_level_list[" << i << "].first = " << log_level_list[i].first;
      } else if (log_level_list[i].first == "DEBUG") {
        EXPECT_EQ(GetSeverity(), Severity::DEBUG)
            << "log_level_list[" << i << "].first = " << log_level_list[i].first;
      } else if (log_level_list[i].first == "VERBOSE") {
        EXPECT_EQ(GetSeverity(), Severity::VERBOSE)
            << "log_level_list[" << i << "].first = " << log_level_list[i].first;
      } else if (log_level_list[i].first == "NONE") {
        EXPECT_EQ(GetSeverity(), Severity::NONE)
            << "log_level_list[" << i << "].first = " << log_level_list[i].first;
      } else if (log_level_list[i].first == "ALL") {
        EXPECT_EQ(GetSeverity(), Severity::ALL)
            << "log_level_list[" << i << "].first = " << log_level_list[i].first;
      }
    }
  }

  EnvVarWrapper wrapper("GXF_LOG_LEVEL_TEST", "DEBUG");
  // Set severity from environment variable.
  SetSeverityFromEnv("GXF_LOG_LEVEL_TEST");

  testing::internal::CaptureStdout();  // WARNING/INFO/DEBUG/VERBOSE are written to stdout
  testing::internal::CaptureStderr();  // ERROR/PANIC are written to stderr

  GXF_LOG_PANIC("PANIC");
  GXF_LOG_ERROR("ERROR");
  GXF_LOG_WARNING("WARNING");
  GXF_LOG_INFO("INFO");
  GXF_LOG_DEBUG("DEBUG");
  GXF_LOG_VERBOSE("VERBOSE");

  std::string log_stdout = testing::internal::GetCapturedStdout();
  std::string log_stderr = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(log_stderr.find("PANIC") != std::string::npos);
  EXPECT_TRUE(log_stderr.find("ERROR") != std::string::npos);
  EXPECT_TRUE(log_stdout.find("WARNING") != std::string::npos);
  EXPECT_TRUE(log_stdout.find("INFO") != std::string::npos);
  EXPECT_TRUE(log_stdout.find("DEBUG") != std::string::npos);
  // VERBOSE would not be printed because the severity is set to DEBUG.
  EXPECT_TRUE(log_stdout.find("VERBOSE") == std::string::npos);
}

TEST(TestLogger, TestGetSeverityFromEnv) {
  // Restore the log severity to Info after the test.
  struct SetSeverityCleanup {
    ~SetSeverityCleanup() { SetSeverity(Severity::INFO); }
  } set_severity_cleanup;

  // Set the default logging function.
  LoggingFunction = DefaultConsoleLogging;

  const std::pair<std::string, std::pair<Severity, int>> log_level_list[] = {
      {"PANIC", {Severity::COUNT, 1}},     {"ERROR", {Severity::ERROR, 0}},
      {"WARNING", {Severity::WARNING, 0}}, {"INFO", {Severity::INFO, 0}},
      {"DEBUG", {Severity::DEBUG, 0}},     {"VERBOSE", {Severity::VERBOSE, 0}},
      {"NONE", {Severity::NONE, 0}},       {"ALL", {Severity::COUNT, 1}},
      {"invalid", {Severity::COUNT, 1}},   {"", {Severity::COUNT, 0}},
  };

  // Test all possible log levels with log_level_list
  for (size_t i = 0; i < sizeof(log_level_list) / sizeof(log_level_list[0]); i++) {
    // Set the environment variable temporarily.
    // It will be restored when the wrapper goes out of scope.
    EnvVarWrapper wrapper(kGxfLogEnvName, log_level_list[i].first.c_str());
    // Get severity from environment variable.
    int error_code = 0;
    Severity severity = GetSeverityFromEnv(kGxfLogEnvName, &error_code);

    // Check if the severity is set correctly.
    EXPECT_EQ(severity, log_level_list[i].second.first)
        << "log_level_list[" << i << "].first = " << log_level_list[i].first;
    EXPECT_EQ(error_code, log_level_list[i].second.second)
        << "log_level_list[" << i << "].first = " << log_level_list[i].first;
  }
}

}  // namespace test
}  // namespace nvidia
