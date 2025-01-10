/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

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

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

}  // namespace

class GxfParameterGetFloat32_Test : public ::testing::Test {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
    GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::BlockMemoryPool", &tid));
    GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "test", &cid));
  }

  void TearDown() {
    GXF_ASSERT_SUCCESS(GxfComponentRemoveWithUID(context, cid));
    GXF_ASSERT_SUCCESS(GxfEntityDestroy(context, eid));
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
  }

 protected:
  gxf_context_t context = kNullContext;
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  gxf_uid_t eid = kNullUid;
  const GxfEntityCreateInfo entity_create_info = {0};
  gxf_tid_t tid = GxfTidNull();
  gxf_uid_t cid = kNullUid;
  float value;
};

TEST_F(GxfParameterGetFloat32_Test, ValidParameter) {
  float float32_var = 101.1;
  GXF_ASSERT_SUCCESS(GxfParameterSetFloat32(context, cid, "float32_var", float32_var));

  GXF_ASSERT_SUCCESS(GxfParameterGetFloat32(context, cid, "float32_var", &value));
  GXF_ASSERT_EQ(value, float32_var);
}

TEST_F(GxfParameterGetFloat32_Test, InvalidContext) {
  GXF_ASSERT_EQ(GxfParameterGetFloat32(kNullContext, cid, "float32_var", &value),
                GXF_CONTEXT_INVALID);
}

TEST_F(GxfParameterGetFloat32_Test, InvalidUid) {
  GXF_ASSERT_EQ(GxfParameterGetFloat32(context, kNullUid, "float32_var", &value),
                GXF_PARAMETER_NOT_FOUND);
}

TEST_F(GxfParameterGetFloat32_Test, InvalidKey) {
  float float32_var = 101.1;
  GXF_ASSERT_SUCCESS(GxfParameterSetFloat32(context, cid, "float32_var", float32_var));
  GXF_ASSERT_EQ(GxfParameterGetFloat32(context, cid, "float32_invalid_var", &value),
                GXF_PARAMETER_NOT_FOUND);
}

TEST_F(GxfParameterGetFloat32_Test, InvalidParameterType) {
  int64_t int64_var = 101;
  GXF_ASSERT_SUCCESS(GxfParameterSetInt64(context, cid, "int64_var", int64_var));
  GXF_ASSERT_EQ(GxfParameterGetFloat32(context, cid, "int64_var", &value),
                GXF_PARAMETER_INVALID_TYPE);
}
