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

class GxfExtensionInfoTest : public ::testing::Test {
 protected:
  void SetUp() override {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info_1));
    info.num_extensions = 1;
    extensions.resize(info.num_extensions);
    info.extensions = extensions.data();
    GXF_ASSERT_EQ(GxfRuntimeInfo(context, &info), GXF_SUCCESS);
  }

  gxf_context_t context = kNullContext;
  const GxfLoadExtensionsInfo info_1{&kStdExtensionFilename, 1, nullptr, 0, nullptr};
  gxf_runtime_info info;
  std::vector<gxf_tid_t> extensions;
};

TEST_F(GxfExtensionInfoTest, InvalidContext) {
  gxf_tid_t libext_std_tid = info.extensions[0];
  gxf_extension_info_t ext_info;
  GXF_ASSERT_EQ(GxfExtensionInfo(nullptr, libext_std_tid, &ext_info), GXF_CONTEXT_INVALID);
}

TEST_F(GxfExtensionInfoTest, NullExtensionInfoPointer) {
  gxf_tid_t libext_std_tid = info.extensions[0];
  GXF_ASSERT_EQ(GxfExtensionInfo(context, libext_std_tid, nullptr), GXF_NULL_POINTER);
}

TEST_F(GxfExtensionInfoTest, InvalidComponentCount) {
  gxf_extension_info_t ext_info;
  gxf_tid_t libext_std_tid = info.extensions[0];
  std::vector<gxf_tid_t> component_tid_list(1);
  ext_info.num_components = 1;
  ext_info.components = component_tid_list.data();
  GXF_ASSERT_EQ(GxfExtensionInfo(context, libext_std_tid, &ext_info), GXF_SUCCESS);
}

TEST_F(GxfExtensionInfoTest, IncorrectComponentTid) {
  gxf_extension_info_t ext_info;
  gxf_tid_t libext_std_tid = info.extensions[0];
  std::vector<gxf_tid_t> component_tid_list(128);
  ext_info.num_components = 128;
  ext_info.components = component_tid_list.data();
  libext_std_tid.hash1 = 0x0c3c0ec777f14312;
  GXF_ASSERT_EQ(GxfExtensionInfo(context, libext_std_tid, &ext_info), GXF_EXTENSION_NOT_FOUND);
}

TEST_F(GxfExtensionInfoTest, Valid) {
  gxf_extension_info_t ext_info;
  gxf_tid_t libext_std_tid = info.extensions[0];
  std::vector<gxf_tid_t> component_tid_list(128);
  ext_info.num_components = 128;
  ext_info.components = component_tid_list.data();
  GXF_ASSERT_EQ(GxfExtensionInfo(context, libext_std_tid, &ext_info), GXF_SUCCESS);
  GXF_ASSERT_EQ(strcmp(ext_info.name, "StandardExtension"), 0);
  GXF_ASSERT_EQ(strcmp(ext_info.author, "NVIDIA"), 0);
}