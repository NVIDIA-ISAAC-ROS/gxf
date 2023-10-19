/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"

namespace nvidia {
namespace gxf {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

TEST(TestEntityRecordReplay, Record) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &info), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphLoadFile(context, "gxf/serialization/tests/test_entity_recorder.yaml"),
            GXF_SUCCESS);
  ASSERT_EQ(GxfGraphActivate(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphRunAsync(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphWait(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphDeactivate(context), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(TestEntityRecordReplay, Replay) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &info), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphLoadFile(context, "gxf/serialization/tests/test_entity_replayer.yaml"),
            GXF_SUCCESS);
  ASSERT_EQ(GxfGraphActivate(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphRunAsync(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphWait(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphDeactivate(context), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

}  // namespace gxf
}  // namespace nvidia
