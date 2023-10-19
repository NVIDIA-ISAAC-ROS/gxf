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

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

}  // namespace

class GxfParameterSetInt32_Test : public ::testing::Test {
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
  int32_t value;
};

TEST_F(GxfParameterSetInt32_Test, ValidParameter) {
  int32_t int32_var = 184;
  GXF_ASSERT_SUCCESS(GxfParameterSetInt32(context, cid, "int32_var", int32_var));
  GXF_ASSERT_SUCCESS(GxfParameterGetInt32(context, cid, "int32_var", &value));
  GXF_ASSERT_EQ(value, int32_var);
}

TEST_F(GxfParameterSetInt32_Test, InvalidContext) {
  int32_t int32_var = 184;
  GXF_ASSERT_EQ(GxfParameterSetInt32(kNullContext, cid, "int32_var", int32_var),
                GXF_CONTEXT_INVALID);
}

TEST_F(GxfParameterSetInt32_Test, OverwritingValidParameterType) {
  int32_t int32_var_1 = 184;
  int32_t int32_var_2 = 121;
  GXF_ASSERT_SUCCESS(GxfParameterSetInt32(context, cid, "int32_var", int32_var_1));
  GXF_ASSERT_SUCCESS(GxfParameterGetInt32(context, cid, "int32_var", &value));
  GXF_ASSERT_EQ(value, int32_var_1);
  GXF_ASSERT_SUCCESS(GxfParameterSetInt32(context, cid, "int32_var", int32_var_2));
  GXF_ASSERT_SUCCESS(GxfParameterGetInt32(context, cid, "int32_var", &value));
  GXF_ASSERT_EQ(value, int32_var_2);
}

TEST_F(GxfParameterSetInt32_Test, SettingInvalidParameterType) {
  int32_t int32_var = 129;
  double float64_var = 101.1;
  GXF_ASSERT_SUCCESS(GxfParameterSetFloat64(context, cid, "float64_var", float64_var));
  GXF_ASSERT_EQ(GxfParameterSetInt32(context, cid, "float64_var", int32_var),
                GXF_PARAMETER_INVALID_TYPE);
}

TEST_F(GxfParameterSetInt32_Test, LowerBoundryValue) {
  GXF_ASSERT_SUCCESS(GxfParameterSetInt32(context, cid, "int32_var", INT32_MIN));
  GXF_ASSERT_SUCCESS(GxfParameterGetInt32(context, cid, "int32_var", &value));
  GXF_ASSERT_EQ(value, INT32_MIN);
}

TEST_F(GxfParameterSetInt32_Test, UpperBoundryValue) {
  GXF_ASSERT_SUCCESS(GxfParameterSetInt32(context, cid, "int32_var", INT32_MAX));
  GXF_ASSERT_SUCCESS(GxfParameterGetInt32(context, cid, "int32_var", &value));
  GXF_ASSERT_EQ(value, INT32_MAX);
}
