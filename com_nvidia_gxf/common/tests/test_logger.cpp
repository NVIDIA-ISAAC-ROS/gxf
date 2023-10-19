/*
Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gtest/gtest.h"
#include <cstdint>

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

}  // namespace test
}  // namespace nvidia
