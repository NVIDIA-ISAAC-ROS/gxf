/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <cstdint>
#include <cstdlib>

#include "gxf/logger/gxf_logger.hpp"
#include "gtest/gtest.h"

namespace nvidia {
namespace logger {
namespace test {

class TestGxfLogger : public GxfLogger {
 public:
  static std::shared_ptr<TestGxfLogger> create() { return std::make_shared<TestGxfLogger>(); }

  TestGxfLogger() : GxfLogger() {
    // Save the current severity and sinks
    old_severity_ = level();
    for (int severity = 0; severity < static_cast<int>(Severity::COUNT); ++severity) {
      old_sinks_[severity] = redirect(severity);
    }
  }

  ~TestGxfLogger() {
    // Restore the old severity and sinks
    level(old_severity_);
    for (int severity = 0; severity < static_cast<int>(Severity::COUNT); ++severity) {
      redirect(severity, old_sinks_[severity]);
    }
  }

  int old_severity_ = static_cast<int>(Severity::INFO);
  void* old_sinks_[static_cast<int>(Severity::COUNT)] = {nullptr};
};

class MockLogger : public ILogger {
 public:
  void log(const char* file, int line, const char* name, int level, const char* message,
           void* arg = nullptr) override {
    log_file_ = file;
    log_line_ = line;
    log_name_ = name;
    log_level_ = level;
    log_message_ = message;
    log_arg_ = arg;
  }
  void expect_log(const char* file, int line, const char* name, int level, const char* message,
                  void* arg = nullptr) {
    ASSERT_STREQ(log_file_, file);
    ASSERT_EQ(log_line_, line);
    ASSERT_STREQ(log_name_, name);
    ASSERT_EQ(log_level_, level);
    ASSERT_STREQ(log_message_, message);
    ASSERT_EQ(log_arg_, arg);
  }

  void pattern(const char* pattern) override { pattern_ = pattern; }
  const char* pattern() const override { return pattern_; }

  void level(int level) override { level_ = level; }
  int level() const override { return level_; }

  void redirect(int level, void* file) override {
    redirect_level_ = level;
    redirect_file_ = file;
  }
  void* redirect(int level) const override {
    return redirect_level_ == level ? redirect_file_ : nullptr;
  }

  const char* log_file_ = nullptr;
  int log_line_ = 0;
  const char* log_name_ = nullptr;
  int log_level_ = 0;
  const char* log_message_ = nullptr;
  void* log_arg_ = nullptr;

  const char* pattern_ = nullptr;

  int level_ = 0;

  int redirect_level_ = 0;
  void* redirect_file_ = nullptr;
};

TEST(TestGxfLogger, SetLogFunction) {
  const char* test_arg = "test_arg";
  bool log_function_called = false;
  auto logger = TestGxfLogger::create();
  auto log_function = [&log_function_called, &test_arg](const char* file, int line,
                                                        const char* name, int level,
                                                        const char* message, void* arg) {
    log_function_called = true;
    ASSERT_STREQ(file, "file.cpp");
    ASSERT_EQ(line, 42);
    ASSERT_STREQ(name, "TestGxfLogger");
    ASSERT_EQ(level, 1);
    ASSERT_STREQ(message, "Log message");
    ASSERT_EQ(arg, test_arg);
  };
  logger->func(log_function, reinterpret_cast<void*>(const_cast<char*>(test_arg)));

  logger->log("file.cpp", 42, "TestGxfLogger", 1, "Log message");

  ASSERT_TRUE(log_function_called);

  const char* pattern = "%Y-%m-%d %H:%M:%S";
  logger->pattern(pattern);
  ASSERT_STREQ("", logger->pattern());  // gxf logger doesn't support pattern

  int level = 2;
  logger->level(level);
  ASSERT_EQ(level, logger->level());

  logger->redirect(level, stderr);
  ASSERT_EQ(stderr, logger->redirect(level));
}

TEST(TestGxfLogger, LoggerInterface) {
  auto logger = TestGxfLogger::create();

  // Create a mock logger object for testing
  auto mock_logger = std::make_shared<MockLogger>();
  logger->logger(mock_logger);

  logger->log("file2.cpp", 72, "TestLogger2", 2, "Log message2");
  mock_logger->expect_log("file2.cpp", 72, "TestLogger2", 2, "Log message2", nullptr);

  const char* pattern = "%Y-%m-%d %H:%M:%S";
  logger->pattern(pattern);
  ASSERT_STREQ(pattern, logger->pattern());
  ASSERT_STREQ(pattern, mock_logger->pattern());

  int level = 2;
  logger->level(level);
  ASSERT_EQ(level, logger->level());
  ASSERT_EQ(level, mock_logger->level());

  logger->redirect(level, stdout);
  ASSERT_EQ(stdout, logger->redirect(level));
  ASSERT_EQ(stdout, mock_logger->redirect(level));
}

TEST(TestGxfLogger, PreferLogFunction) {
  const char* test_arg = "test_arg";
  bool log_function_called = false;
  auto logger = TestGxfLogger::create();
  auto log_function = [&log_function_called, &test_arg](const char* file, int line,
                                                        const char* name, int level,
                                                        const char* message, void* arg) {
    log_function_called = true;
    ASSERT_STREQ(file, "file.cpp");
    ASSERT_EQ(line, 42);
    ASSERT_STREQ(name, "TestGxfLogger");
    ASSERT_EQ(level, 1);
    ASSERT_STREQ(message, "Log message");
    ASSERT_EQ(arg, test_arg);
  };
  logger->func(log_function, reinterpret_cast<void*>(const_cast<char*>(test_arg)));

  // Create a mock logger object for testing
  auto mock_logger = std::make_shared<MockLogger>();

  logger->logger(mock_logger);

  // Call log function. Expect that the log function is called and not the mock logger.
  logger->log("file.cpp", 42, "TestGxfLogger", 1, "Log message");

  ASSERT_TRUE(log_function_called);

  // No logger interface calls should be made
  mock_logger->expect_log(nullptr, 0, nullptr, 0, nullptr, nullptr);

  // For pattern, level and redirect, the logger interface should be used
  const char* pattern = "%Y-%m-%d %H:%M:%S";
  logger->pattern(pattern);
  ASSERT_STREQ(pattern, logger->pattern());
  ASSERT_STREQ(pattern, mock_logger->pattern());

  int level = 2;
  logger->level(level);
  ASSERT_EQ(level, logger->level());
  ASSERT_EQ(level, mock_logger->level());

  logger->redirect(level, stdout);
  ASSERT_EQ(stdout, logger->redirect(level));
  ASSERT_EQ(stdout, mock_logger->redirect(level));
}

TEST(TestGxfLogger, NoRegistration) {
  auto logger = TestGxfLogger::create();

  // Do nothing
  logger->log("file.cpp", 42, "TestGxfLogger", 1, "Log message");

  const char* pattern = "%Y-%m-%d %H:%M:%S";
  logger->pattern(pattern);
  ASSERT_STREQ("", logger->pattern());  // gxf logger doesn't support pattern

  int level = 5;
  logger->level(level);
  ASSERT_EQ(level, logger->level());

  logger->redirect(level, stderr);
  ASSERT_EQ(stderr, logger->redirect(level));
  // "constexpr int kMaxSeverity = 255;" in logger.cpp
  ASSERT_DEATH(logger->redirect(256, stderr), "");  // asserts that abort() is called
}

}  // namespace test
}  // namespace logger
}  // namespace nvidia
