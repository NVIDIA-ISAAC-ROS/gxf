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

class GxfGraphInterruptTest : public ::testing::Test {
 protected:
  void SetUp() override {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info_std));
  }

  gxf_context_t context = kNullContext;
  const GxfLoadExtensionsInfo info_std{kExtensions, 3, nullptr, 0, nullptr};
};

TEST_F(GxfGraphInterruptTest, InvalidContext) {
  GXF_ASSERT_EQ(GxfGraphInterrupt(nullptr), GXF_CONTEXT_INVALID);
}

TEST_F(GxfGraphInterruptTest, InterruptBeforeActivate) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_EQ(GxfGraphInterrupt(context), GXF_INVALID_EXECUTION_SEQUENCE);
}

TEST_F(GxfGraphInterruptTest, InterruptAfterActivate) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_EQ(GxfGraphInterrupt(context), GXF_INVALID_EXECUTION_SEQUENCE);
}

TEST_F(GxfGraphInterruptTest, InterruptAfterRun) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRun(context));
  GXF_ASSERT_EQ(GxfGraphInterrupt(context), GXF_INVALID_EXECUTION_SEQUENCE);
}

TEST_F(GxfGraphInterruptTest, InterruptAfterRunAsync) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphInterrupt(context));
}

TEST_F(GxfGraphInterruptTest, InterruptAfterWait) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  GXF_ASSERT_EQ(GxfGraphInterrupt(context), GXF_INVALID_EXECUTION_SEQUENCE);
}

TEST_F(GxfGraphInterruptTest, InterruptAfterDeactivate) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRun(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_EQ(GxfGraphInterrupt(context), GXF_INVALID_EXECUTION_SEQUENCE);
}

TEST_F(GxfGraphInterruptTest, ConsecutiveInterrupts) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphWait.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphInterrupt(context));
  GXF_ASSERT_EQ(GxfGraphInterrupt(context), GXF_INVALID_EXECUTION_SEQUENCE);
}