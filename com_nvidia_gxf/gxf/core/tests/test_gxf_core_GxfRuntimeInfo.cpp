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
constexpr const char* kStdExtensionFilename = "gxf/std/libgxf_std.so";
}  // namespace

class GxfRuntimeInfoTest : public ::testing::Test {
 protected:
  void SetUp() override {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info_1));
  }

  gxf_context_t context = kNullContext;
  const GxfLoadExtensionsInfo info_1{&kStdExtensionFilename, 1, nullptr, 0, nullptr};
};

TEST_F(GxfRuntimeInfoTest, InvalidContext) {
  gxf_runtime_info info;
  info.num_extensions = 1;
  GXF_ASSERT_EQ(GxfRuntimeInfo(kNullContext, &info), GXF_CONTEXT_INVALID);
}

TEST_F(GxfRuntimeInfoTest, InvalidRuntimeInfoPointer) {
  GXF_ASSERT_EQ(GxfRuntimeInfo(context, nullptr), GXF_NULL_POINTER);
}

TEST_F(GxfRuntimeInfoTest, InvalidExtensionCount) {
  gxf_runtime_info info;
  info.num_extensions = 0;
  GXF_ASSERT_EQ(GxfRuntimeInfo(context, &info), GXF_RESULT_ARRAY_TOO_SMALL);
}

TEST_F(GxfRuntimeInfoTest, Valid) {
  gxf_runtime_info info;
  info.num_extensions = 1;
  std::vector<gxf_tid_t> extensions;
  extensions.resize(info.num_extensions);
  info.extensions = extensions.data();
  GXF_ASSERT_SUCCESS(GxfRuntimeInfo(context, &info));

  gxf_runtime_info rt_info;
  rt_info.num_extensions = 2;
  extensions.resize(2);
  rt_info.extensions = extensions.data();
  GXF_ASSERT_SUCCESS(GxfRuntimeInfo(context, &rt_info));
  GXF_ASSERT_EQ(rt_info.num_extensions, 1);
}

TEST_F(GxfRuntimeInfoTest, MultipleExtensions) {
  const char* kTestExtensionFilename = "gxf/test/extensions/libgxf_test.so";
  const GxfLoadExtensionsInfo info_2{&kTestExtensionFilename, 1, nullptr, 0, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info_2));

  gxf_runtime_info info;
  info.num_extensions = 2;
  std::vector<gxf_tid_t> extensions;
  extensions.resize(info.num_extensions);
  info.extensions = extensions.data();
  GXF_ASSERT_SUCCESS(GxfRuntimeInfo(context, &info));
  GXF_ASSERT_EQ(info.extensions[1].hash1, 0x1b99ffebc2504ced);
}

TEST_F(GxfRuntimeInfoTest, ExtensionWithNullHash1) {
  const char* kTestExtensionFilename = "gxf/core/tests/test_extension/libgxf_test_extension.so";
  const GxfLoadExtensionsInfo info_2{&kTestExtensionFilename, 1, nullptr, 0, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info_2));

  gxf_runtime_info info;
  info.num_extensions = 2;
  std::vector<gxf_tid_t> extensions;
  extensions.resize(info.num_extensions);
  info.extensions = extensions.data();
  GXF_ASSERT_EQ(GxfRuntimeInfo(context, &info), GXF_SUCCESS);
  GXF_ASSERT_EQ(info.extensions[1].hash1, kNullUid);
}
