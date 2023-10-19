/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"

namespace {
const char* kStdExtensionFilename = "gxf/std/libgxf_std.so";
const char* kTestExtensionFilename = "gxf/test/extensions/libgxf_test.so";
const char* kSampleExtensionFilename = "gxf/sample/libgxf_sample.so";
}  // namespace

TEST(GxfGraphSetRootPathTest, ValidRootPath) {
  gxf_context_t context = kNullContext;
  const char* kGraphFileName = "tests/apps/test_app_GxfGraphLoadFile_valid.yaml";
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  const GxfLoadExtensionsInfo info_std{&kStdExtensionFilename, 1, nullptr, 0, nullptr};
  const GxfLoadExtensionsInfo info_test{&kTestExtensionFilename, 1, nullptr, 0, nullptr};
  const GxfLoadExtensionsInfo info_sample{&kSampleExtensionFilename, 1, nullptr, 0, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info_std));
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info_sample));
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info_test));

  GXF_ASSERT_SUCCESS(GxfGraphSetRootPath(context, "gxf/core"));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));
}

TEST(GxfGraphSetRootPathTest, InvalidContext) {
  GXF_ASSERT_EQ(GxfGraphSetRootPath(nullptr, "gxf/core"), GXF_CONTEXT_INVALID);
}

TEST(GxfGraphSetRootPathTest, NullRootPath) {
  gxf_context_t context = kNullContext;
  GXF_ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
  GXF_ASSERT_EQ(GxfGraphSetRootPath(context, nullptr), GXF_ARGUMENT_NULL);
}

TEST(GxfGraphSetRootPathTest, IncorrectRootPath) {
  gxf_context_t context = kNullContext;
  const char* kGraphFileName = "tests/apps/test_app_GxfGraphLoadFile_valid.yaml";
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  const GxfLoadExtensionsInfo info_std{&kStdExtensionFilename, 1, nullptr, 0, nullptr};
  const GxfLoadExtensionsInfo info_test{&kTestExtensionFilename, 1, nullptr, 0, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info_std));
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info_test));

  GXF_ASSERT_SUCCESS(GxfGraphSetRootPath(context, "gxf/core_incorrect"));
  GXF_ASSERT_EQ(GxfGraphLoadFile(context, kGraphFileName), GXF_FAILURE);
}

TEST(GxfGraphSetRootPathTest, SetRootPathMultipleTimes) {
  gxf_context_t context = kNullContext;
  const char* kGraphFileName = "tests/apps/test_app_GxfGraphLoadFile_valid.yaml";
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  const GxfLoadExtensionsInfo info_std{&kStdExtensionFilename, 1, nullptr, 0, nullptr};
  const GxfLoadExtensionsInfo info_test{&kTestExtensionFilename, 1, nullptr, 0, nullptr};
  const GxfLoadExtensionsInfo info_sample{&kSampleExtensionFilename, 1, nullptr, 0, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info_std));
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info_test));
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info_sample));

  GXF_ASSERT_SUCCESS(GxfGraphSetRootPath(context, "gxf/core/"));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGraphFileName));

  GXF_ASSERT_SUCCESS(GxfGraphSetRootPath(context, "gxf/core_incorrect"));
  GXF_ASSERT_EQ(GxfGraphLoadFile(context, kGraphFileName), GXF_FAILURE);
}