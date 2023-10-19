/*
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"
#include <string>

namespace {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

}  // namespace

class GxfCreateEntityGroup_Test : public ::testing::Test {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context_));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context_, &info_));
  }

  void TearDown() {
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context_));
  }

 protected:
  gxf_context_t context_ = kNullContext;
  const GxfLoadExtensionsInfo info_{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  gxf_uid_t gid = kNullUid;
};

TEST_F(GxfCreateEntityGroup_Test, ValidParameters) {
  GXF_ASSERT_SUCCESS(GxfCreateEntityGroup(context_, "group0", &gid));
}

TEST_F(GxfCreateEntityGroup_Test, ValidParameterNullName) {
  GXF_ASSERT_SUCCESS(GxfCreateEntityGroup(context_, nullptr, &gid));
}

TEST_F(GxfCreateEntityGroup_Test, InvalidParameterNullGid) {
  GXF_ASSERT_EQ(GxfCreateEntityGroup(context_, "group0", nullptr), GXF_ARGUMENT_NULL);
}