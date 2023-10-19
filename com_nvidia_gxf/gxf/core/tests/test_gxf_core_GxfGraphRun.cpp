/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <string.h>
#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"

namespace {
constexpr const char* kExtensions[] = {"gxf/std/libgxf_std.so",
                                       "gxf/sample/libgxf_sample.so",
                                       "gxf/test/extensions/libgxf_test.so"};
}  // namespace

class GxfGraphRunTest : public ::testing::Test {
 protected:
  void SetUp() override {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info_std));
  }

  gxf_context_t context = kNullContext;
  const GxfLoadExtensionsInfo info_std{kExtensions, 3, nullptr, 0, nullptr};
};

TEST_F(GxfGraphRunTest, InvalidContextRun) {
  GXF_ASSERT_EQ(GxfGraphRun(nullptr), GXF_CONTEXT_INVALID);
}

TEST_F(GxfGraphRunTest, RunBeforeActivate) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_EQ(GxfGraphRun(context), GXF_INVALID_EXECUTION_SEQUENCE);
}

TEST_F(GxfGraphRunTest, RunAfterActivate) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRun(context));
}

TEST_F(GxfGraphRunTest, ConsecutiveRuns) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRun(context));
  GXF_ASSERT_EQ(GxfGraphRun(context), GXF_INVALID_LIFECYCLE_STAGE);
}

TEST_F(GxfGraphRunTest, RunAfterRunAsync) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_EQ(GxfGraphRun(context), GXF_INVALID_EXECUTION_SEQUENCE);
}

TEST_F(GxfGraphRunTest, RunAfterWait) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  GXF_ASSERT_EQ(GxfGraphRun(context), GXF_INVALID_LIFECYCLE_STAGE);
}

TEST_F(GxfGraphRunTest, RunAfterDeactivate) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRun(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_EQ(GxfGraphRun(context), GXF_INVALID_EXECUTION_SEQUENCE);
}

TEST_F(GxfGraphRunTest, RunAfterInterrupt) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphInterrupt(context));
  GXF_ASSERT_EQ(GxfGraphRun(context), GXF_INVALID_EXECUTION_SEQUENCE);
}

TEST_F(GxfGraphRunTest, ValidConsecutiveRuns) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRun(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRun(context));
}

TEST_F(GxfGraphRunTest, InvalidContextAsync) {
  GXF_ASSERT_EQ(GxfGraphRunAsync(nullptr), GXF_CONTEXT_INVALID);
}

TEST_F(GxfGraphRunTest, RunAsyncBeforeActivate) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_EQ(GxfGraphRunAsync(context), GXF_INVALID_EXECUTION_SEQUENCE);
}

TEST_F(GxfGraphRunTest, RunAsyncAfterActivate) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
}

TEST_F(GxfGraphRunTest, ConsecutiveRunAyncs) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_EQ(GxfGraphRunAsync(context), GXF_INVALID_EXECUTION_SEQUENCE);
}

TEST_F(GxfGraphRunTest, RunAsyncAfterRun) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRun(context));
  GXF_ASSERT_EQ(GxfGraphRunAsync(context), GXF_INVALID_LIFECYCLE_STAGE);
}

TEST_F(GxfGraphRunTest, RunAsyncAfterWait) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  GXF_ASSERT_EQ(GxfGraphRunAsync(context), GXF_INVALID_LIFECYCLE_STAGE);
}

TEST_F(GxfGraphRunTest, RunAsyncAfterDeactivate) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRun(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_EQ(GxfGraphRunAsync(context), GXF_INVALID_EXECUTION_SEQUENCE);
}

TEST_F(GxfGraphRunTest, RunAsyncAfterInterrupt) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphInterrupt(context));
  GXF_ASSERT_EQ(GxfGraphRunAsync(context), GXF_INVALID_EXECUTION_SEQUENCE);
}

TEST_F(GxfGraphRunTest, ValidConsecutiveRunsAsync) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
}

TEST_F(GxfGraphRunTest, GraphWithNoEntities) {
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRun(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
}

TEST_F(GxfGraphRunTest, InvalidAppRun) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphRun_step_count_mismatch.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRun(context));
  GXF_ASSERT_EQ(GxfGraphDeactivate(context), GXF_FAILURE);
}

TEST_F(GxfGraphRunTest, InvalidAppRunAsync) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphRun_step_count_mismatch.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  GXF_ASSERT_EQ(GxfGraphDeactivate(context), GXF_FAILURE);
}