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

class GxfGraphWaitTest : public ::testing::Test {
 protected:
  void SetUp() override {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info_std));
  }

  gxf_context_t context = kNullContext;
  const GxfLoadExtensionsInfo info_std{kExtensions, 3, nullptr, 0, nullptr};
};

TEST_F(GxfGraphWaitTest, InvalidContext) {
  GXF_ASSERT_EQ(GxfGraphWait(nullptr), GXF_CONTEXT_INVALID);
}

TEST_F(GxfGraphWaitTest, WaitBeforeActivate) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
}

TEST_F(GxfGraphWaitTest, WaitAfterActivate) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
}

TEST_F(GxfGraphWaitTest, WaitAfterRun) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRun(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
}

TEST_F(GxfGraphWaitTest, WaitAfterRunAsync) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
}

TEST_F(GxfGraphWaitTest, WaitAfterInterrupt) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphInterrupt(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
}

TEST_F(GxfGraphWaitTest, WaitAfterDeactivate) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRun(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
}

TEST_F(GxfGraphWaitTest, ConsecutiveWaits) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRun(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
}