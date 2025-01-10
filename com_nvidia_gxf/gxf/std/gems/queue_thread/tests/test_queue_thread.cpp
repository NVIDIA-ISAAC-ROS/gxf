/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/gems/queue_thread/queue_thread.hpp"

#include <thread>

#include "gtest/gtest.h"

namespace nvidia {
namespace gxf {

class QueueThread_Test : public ::testing::Test {
 public:
  using GxfSystemThread = QueueThread<std::string>;
  void SetUp() {
    thread_ = std::make_unique<GxfSystemThread>(
    std::bind(&QueueThread_Test::asyncRunnerCallback, this, std::placeholders::_1, this),
    name());
  }

  void TearDown() {
  }

 protected:
  std::unique_ptr<GxfSystemThread> thread_;
  std::vector<std::string> executed_;
  bool asyncRunnerCallback(std::string event, QueueThread_Test* self) {
    // GXF use Expected<> to replace exception,
    // however thread callback has to add exception catch, the same to std::thread entrance
    bool result = false;
    try {
      GXF_LOG_INFO("QueueThread_Test[%s] thread received event: %s", self->name().c_str(), event.c_str());
      if (event == Event::kMockEventA) {
        result = self->onMockEventA();
      } else if (event == Event::kMockEventB) {
        result = self->onMockEventB();
      } else {
        // an error occurred, clean up if needed then exit
        GXF_LOG_ERROR("Unknown event: %s", event.c_str());
        result = false;
      }

      EXPECT_EQ(result, true);
      return true;
    } catch (...) {
      GXF_LOG_ERROR("QueueThread_Test:%s unexpected error in asyncRunnerCallback.", name().c_str());
      EXPECT_EQ(result, true);
      return false;  // unblock wait() thread
    }
  }

  std::string name() {
    return "UnitTest";
  }

  struct Event {
    static constexpr const char* kMockEventA = "kMockEventA";
    static constexpr const char* kMockEventB = "kMockEventB";
  };
  bool onMockEventA() {
    GXF_LOG_INFO("calling onMockEventA()");
    executed_.push_back(Event::kMockEventA);
    return true;
  }
  bool onMockEventB() {
    GXF_LOG_INFO("calling onMockEventB()");
    executed_.push_back(Event::kMockEventB);
    return true;
  }
  int test_job_count_ = 2 * 8;
  std::future<bool> submitJobs() {
    std::future<bool> resultLast;
    for (int i = 0; i < test_job_count_ / 2; i++) {
      thread_->queueItem(Event::kMockEventA);
      resultLast = thread_->queueItem(Event::kMockEventB);
    }
    return resultLast;
  }
  void verifyExecutedJobs() {
    for (int i = 0; i < test_job_count_ / 2; i++) {
      EXPECT_EQ(executed_.at(i * 2), Event::kMockEventA);
      EXPECT_EQ(executed_.at(i * 2 + 1), Event::kMockEventB);
    }
  }
};

TEST_F(QueueThread_Test, ProcessCompletion) {
  std::future<bool> resultLast = submitJobs();
  // sync on the last
  resultLast.get();
  EXPECT_EQ(executed_.size(), test_job_count_);
  thread_->stop();
}

TEST_F(QueueThread_Test, ProcessOrder) {
  std::future<bool> resultLast = submitJobs();
  // sync on the last
  resultLast.get();
  verifyExecutedJobs();
  thread_->stop();
}

TEST_F(QueueThread_Test, WaitAndStopThreading) {
  // simulate main thread that's blocking on this QueueThread
  bool waitReturn = false;
  std::thread mock_main_thread = std::thread([&] () mutable {
    thread_->wait();
    waitReturn = true;
  });
  // this gTest thread submit jobs and start processing
  std::future<bool> resultLast = submitJobs();
  // sync on the last
  resultLast.get();
  GXF_LOG_INFO("executed_.size() %ld", executed_.size());

  // simulate stopper thread, to stop when jobs are potentially running
  bool stopReturn = false;
  std::thread mock_stopper_thread = std::thread([&] () mutable {
    thread_->stop();
    stopReturn = true;
  });
  // must join stopper thread before wait thread
  if (mock_stopper_thread.joinable()) {
    mock_stopper_thread.join();
  }
  // must join wait thread after stopper thread
  if (mock_main_thread.joinable()) {
    mock_main_thread.join();
  }

  EXPECT_EQ(waitReturn, true);
  EXPECT_EQ(stopReturn, true);
}

}  // namespace gxf
}  // namespace nvidia
