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
#include <mutex>
#include <thread>
#include <vector>

#include "gxf/logger/logger.hpp"
#include "gtest/gtest.h"

namespace nvidia {
namespace logger {
namespace test {

class TestLogger : public Logger {
 public:
  static std::shared_ptr<TestLogger> create() { return std::make_shared<TestLogger>(); }
};

class MockLogger : public ILogger {
 public:
  void log(const char* file, int line, const char* name, int level, const char* message,
           void* arg = nullptr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    log_file_ = file;
    log_line_ = line;
    log_name_ = name;
    log_level_ = level;
    log_message_ = message;
    log_messages_.push_back(message);
    message_count_++;
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

  std::vector<std::string>& log_messages() { return log_messages_; }

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

  int message_count_ = 0;
  std::mutex mutex_;
  std::vector<std::string> log_messages_;

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

TEST(TestCommonLogger, SetLogFunction) {
  const char* test_arg = "test_arg";
  bool log_function_called = false;
  auto logger = TestLogger::create();
  auto log_function = [&log_function_called, &test_arg](const char* file, int line,
                                                        const char* name, int level,
                                                        const char* message, void* arg) {
    log_function_called = true;
    ASSERT_STREQ(file, "file.cpp");
    ASSERT_EQ(line, 42);
    ASSERT_STREQ(name, "TestLogger");
    ASSERT_EQ(level, 1);
    ASSERT_STREQ(message, "Log message");
    ASSERT_EQ(arg, test_arg);
  };
  logger->func(log_function, reinterpret_cast<void*>(const_cast<char*>(test_arg)));

  logger->log("file.cpp", 42, "TestLogger", 1, "Log message");

  ASSERT_TRUE(log_function_called);

  const char* pattern = "%Y-%m-%d %H:%M:%S";
  logger->pattern(pattern);
  ASSERT_STREQ(pattern, logger->pattern());

  int level = 2;
  logger->level(level);
  ASSERT_EQ(level, logger->level());

  logger->redirect(level, stderr);
  ASSERT_EQ(stderr, logger->redirect(level));
}

TEST(TestCommonLogger, LoggerInterface) {
  auto logger = TestLogger::create();

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

TEST(TestCommonLogger, PreferLogFunction) {
  const char* test_arg = "test_arg";
  bool log_function_called = false;
  auto logger = TestLogger::create();
  auto log_function = [&log_function_called, &test_arg](const char* file, int line,
                                                        const char* name, int level,
                                                        const char* message, void* arg) {
    log_function_called = true;
    ASSERT_STREQ(file, "file.cpp");
    ASSERT_EQ(line, 42);
    ASSERT_STREQ(name, "TestLogger");
    ASSERT_EQ(level, 1);
    ASSERT_STREQ(message, "Log message");
    ASSERT_EQ(arg, test_arg);
  };
  logger->func(log_function, reinterpret_cast<void*>(const_cast<char*>(test_arg)));

  // Create a mock logger object for testing
  auto mock_logger = std::make_shared<MockLogger>();

  logger->logger(mock_logger);

  // Call log function. Expect that the log function is called and not the mock logger.
  logger->log("file.cpp", 42, "TestLogger", 1, "Log message");

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

TEST(TestCommonLogger, UseSuppliedFunctionArg) {
  const char* test_arg = "test_arg";
  bool log_function_called = false;
  auto logger = TestLogger::create();
  auto log_function = [&log_function_called, &test_arg](const char* file, int line,
                                                        const char* name, int level,
                                                        const char* message, void* arg) {
    log_function_called = true;
    ASSERT_STREQ(file, "file.cpp");
    ASSERT_EQ(line, 42);
    ASSERT_STREQ(name, "TestLogger");
    ASSERT_EQ(level, 1);
    ASSERT_STREQ(message, "Log message");
    ASSERT_EQ(arg, test_arg);
  };

  // Set the log function with test_arg as the function argument
  logger->func(log_function, reinterpret_cast<void*>(const_cast<char*>(test_arg)));

  // Create a mock logger object for testing
  auto mock_logger = std::make_shared<MockLogger>();

  logger->logger(mock_logger);

  // Call the log function without an argument.
  // Expect that the log function is called and not the mock logger.
  logger->log("file.cpp", 42, "TestLogger", 1, "Log message");

  ASSERT_TRUE(log_function_called);
}

TEST(TestCommonLogger, NoRegistration) {
  auto logger = TestLogger::create();

  // Do nothing
  logger->log("file.cpp", 42, "TestLogger", 1, "Log message");

  const char* pattern = "%Y-%m-%d %H:%M:%S";
  logger->pattern(pattern);
  ASSERT_STREQ(pattern, logger->pattern());

  int level = 5;
  logger->level(level);
  ASSERT_EQ(level, logger->level());

  logger->redirect(level, stderr);
  ASSERT_EQ(stderr, logger->redirect(level));
  ASSERT_EQ(nullptr, logger->redirect(0));
  // "constexpr int kMaxSeverity = 255;" in logger.cpp
  logger->redirect(256, stderr);
  ASSERT_EQ(nullptr, logger->redirect(256));
}

// Test case for logger in a multithreaded environment
TEST(TestCommonLogger, MultithreadedLogging) {
  // TODO: Enhance or add test cases after implementing a thread-safe logger.

  // Create a logger instance
  auto logger = TestLogger::create();

  // Create a mock logger object for testing
  auto mock_logger = std::make_shared<MockLogger>();
  logger->logger(mock_logger);

  // Number of threads
  const int num_threads = 100;

  // Number of log messages per thread
  const int num_messages_per_thread = 1000;

  // Function to log messages
  auto log_messages_func = [&logger](int thread_id) {
    for (int i = 0; i < num_messages_per_thread; ++i) {
      std::string message =
          "Thread " + std::to_string(thread_id) + " - Log message " + std::to_string(i);
      logger->log("file.cpp", 42, "TestLogger", 1, message.c_str());
    }
  };

  // Create and start the threads
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) { threads.emplace_back(log_messages_func, i); }

  // Wait for all threads to finish
  for (auto& thread : threads) { thread.join(); }

  // Verify that all log messages are logged correctly

  // Check the number of log messages
  auto& log_messages = mock_logger->log_messages();
  ASSERT_EQ(num_threads * num_messages_per_thread, static_cast<int>(log_messages.size()));

  // Check if all log messages are properly logged.
  // The order of log messages is not guaranteed.
  for (int i = 0; i < num_threads; ++i) {
    for (int j = 0; j < num_messages_per_thread; ++j) {
      std::string message = "Thread " + std::to_string(i) + " - Log message " + std::to_string(j);
      ASSERT_TRUE(std::find(log_messages.begin(), log_messages.end(), message) !=
                  log_messages.end());
    }
  }
}

}  // namespace test
}  // namespace logger
}  // namespace nvidia
