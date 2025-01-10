/*
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <chrono>
#include <cstring>
#include <thread>
#include <iostream>
#include <fstream>

#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"
#include "gxf/core/entity.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/std/default_extension.hpp"
#include "gxf/std/gems/utils/time.hpp"

// #include "test_load_extension.hpp"

namespace {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";
constexpr const char* kStdExtensionFilename = "gxf/std/libgxf_std.so";

}  // namespace

class EntityCreateRate_Test : public ::testing::Test {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context_));
    const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context_, &info));
  }
  void TearDown() {
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context_));
  }
 protected:
  gxf_context_t context_ = kNullContext;
  const int kEntitiesPerThread = 100000;  // 100k per thread
  const std::vector<int> thread_nums_ = {1, 2, 4, 8};
  const int kSampleTimes = 5;
  long multiThreadCreateEntities(int thread_num, int entities_per_thread);
};

long EntityCreateRate_Test::multiThreadCreateEntities(int thread_num, int entities_per_thread) {
  std::thread t[thread_num];
  // Time to start creating entities_per_thread Entity objects per thread
  long timeBegin = nvidia::gxf::getCurrentTimeUs();
  for (int j = 0; j < thread_num; j++) {
    t[j] = std::thread([&, j] () mutable {
      for (int i = 0; i < entities_per_thread; i++) {
        nvidia::gxf::Entity ent;
        ent = std::move(nvidia::gxf::Entity::New(context_).value());
      }
    });
  }
  for (int j = 0; j < thread_num; j++) {
    t[j].join();
  }
  // Time to finish all threads
  long timeEnd = nvidia::gxf::getCurrentTimeUs();
  long duration = timeEnd - timeBegin;
  GXF_LOG_INFO("**********************TIME TAKEN %ld ms, threadNum: %d\n",
    duration/1000, thread_num);
  return duration;
}

TEST_F(EntityCreateRate_Test, EntityOnly) {
  for (const int& thread_num : thread_nums_) {
    long duration = 0;
    for (int i = 0; i < kSampleTimes; i++) {
      long duration_i = multiThreadCreateEntities(thread_num, kEntitiesPerThread);
      duration += duration_i;
    }
    duration = duration / (1000 * kSampleTimes);
    long entity_num = thread_num * kEntitiesPerThread;
    (void)entity_num;  // avoid unused variable warning
    GXF_LOG_INFO("threadNum %d: Avg TIME TAKEN %ld ms, Throughput %ld Entities/s \n",
      thread_num, duration, entity_num * 1000 / duration);
  }
}