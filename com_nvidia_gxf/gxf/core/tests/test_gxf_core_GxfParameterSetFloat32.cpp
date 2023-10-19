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

namespace {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

}  // namespace

class GxfParameterSetFloat32_Test : public ::testing::Test {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
    GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::BlockMemoryPool", &tid));
    GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "test", &cid));
  }

  void TearDown() {
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

TEST_F(GxfParameterSetFloat32_Test, ValidParameter) {
  float float32_var = 101.1;
  GXF_ASSERT_SUCCESS(GxfParameterSetFloat32(context, cid, "float32_var", float32_var));

  GXF_ASSERT_SUCCESS(GxfParameterGetFloat32(context, cid, "float32_var", &value));
  GXF_ASSERT_EQ(value, float32_var);
}

TEST_F(GxfParameterSetFloat32_Test, InvalidContext) {
  float float32_var = 101.1;
  GXF_ASSERT_EQ(GxfParameterSetFloat32(kNullContext, cid, "float32_var", float32_var),
                GXF_CONTEXT_INVALID);
}

TEST_F(GxfParameterSetFloat32_Test, OverwritingValidParameterType) {
  float float32_var_1 = 129.2;
  float float32_var_2 = 101.1;
  GXF_ASSERT_SUCCESS(GxfParameterSetFloat32(context, cid, "float32_var", float32_var_1));
  GXF_ASSERT_SUCCESS(GxfParameterGetFloat32(context, cid, "float32_var", &value));
  GXF_ASSERT_EQ(value, float32_var_1);
  GXF_ASSERT_SUCCESS(GxfParameterSetFloat32(context, cid, "float32_var", float32_var_2));
  GXF_ASSERT_SUCCESS(GxfParameterGetFloat32(context, cid, "float32_var", &value));
  GXF_ASSERT_EQ(value, float32_var_2);
}

TEST_F(GxfParameterSetFloat32_Test, SettingInvalidParameterType) {
  int64_t int64_var = 129;
  float float32_var = 101.1;
  GXF_ASSERT_SUCCESS(GxfParameterSetInt64(context, cid, "int64_var", int64_var));
  GXF_ASSERT_EQ(GxfParameterSetFloat32(context, cid, "int64_var", float32_var),
                GXF_PARAMETER_INVALID_TYPE);
}

TEST_F(GxfParameterSetFloat32_Test, LowerBoundryValue) {
  GXF_ASSERT_SUCCESS(GxfParameterSetFloat32(context, cid, "float32_var", FLT_MIN));
  GXF_ASSERT_SUCCESS(GxfParameterGetFloat32(context, cid, "float32_var", &value));
  GXF_ASSERT_EQ(value, FLT_MIN);
}

TEST_F(GxfParameterSetFloat32_Test, UpperBoundryValue) {
  GXF_ASSERT_SUCCESS(GxfParameterSetFloat32(context, cid, "float32_var", FLT_MAX));
  GXF_ASSERT_SUCCESS(GxfParameterGetFloat32(context, cid, "float32_var", &value));
  GXF_ASSERT_EQ(value, FLT_MAX);
}

TEST_F(GxfParameterSetFloat32_Test, NegativeLowerBoundryValue) {
  GXF_ASSERT_SUCCESS(GxfParameterSetFloat32(context, cid, "float32_var", -FLT_MIN));
  GXF_ASSERT_SUCCESS(GxfParameterGetFloat32(context, cid, "float32_var", &value));
  GXF_ASSERT_EQ(value, -FLT_MIN);
}

TEST_F(GxfParameterSetFloat32_Test, NegativeUpperBoundryValue) {
  GXF_ASSERT_SUCCESS(GxfParameterSetFloat32(context, cid, "float32_var", -FLT_MAX));
  GXF_ASSERT_SUCCESS(GxfParameterGetFloat32(context, cid, "float32_var", &value));
  GXF_ASSERT_EQ(value, -FLT_MAX);
}
