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

class GxfParameterSetInt64_Test : public ::testing::Test {
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
  int64_t value;
};

TEST_F(GxfParameterSetInt64_Test, ValidParameter) {
  int64_t int64_var = 184;
  GXF_ASSERT_SUCCESS(GxfParameterSetInt64(context, cid, "int64_var", int64_var));
  GXF_ASSERT_SUCCESS(GxfParameterGetInt64(context, cid, "int64_var", &value));
  GXF_ASSERT_EQ(value, int64_var);
}

TEST_F(GxfParameterSetInt64_Test, InvalidContext) {
  int64_t int64_var = 184;
  GXF_ASSERT_EQ(GxfParameterSetInt64(kNullContext, cid, "int64_var", int64_var),
                GXF_CONTEXT_INVALID);
}

TEST_F(GxfParameterSetInt64_Test, OverwritingValidParameterType) {
  int64_t int64_var_1 = 184;
  int64_t int64_var_2 = 121;
  GXF_ASSERT_SUCCESS(GxfParameterSetInt64(context, cid, "int64_var", int64_var_1));
  GXF_ASSERT_SUCCESS(GxfParameterGetInt64(context, cid, "int64_var", &value));
  GXF_ASSERT_EQ(value, int64_var_1);
  GXF_ASSERT_SUCCESS(GxfParameterSetInt64(context, cid, "int64_var", int64_var_2));
  GXF_ASSERT_SUCCESS(GxfParameterGetInt64(context, cid, "int64_var", &value));
  GXF_ASSERT_EQ(value, int64_var_2);
}

TEST_F(GxfParameterSetInt64_Test, SettingInvalidParameterType) {
  int64_t int64_var = 129;
  double float64_var = 101.1;
  GXF_ASSERT_SUCCESS(GxfParameterSetFloat64(context, cid, "float64_var", float64_var));
  GXF_ASSERT_EQ(GxfParameterSetInt64(context, cid, "float64_var", int64_var),
                GXF_PARAMETER_INVALID_TYPE);
}

TEST_F(GxfParameterSetInt64_Test, LowerBoundryValue) {
  GXF_ASSERT_SUCCESS(GxfParameterSetInt64(context, cid, "int64_var", INT64_MIN));
  GXF_ASSERT_SUCCESS(GxfParameterGetInt64(context, cid, "int64_var", &value));
  GXF_ASSERT_EQ(value, INT64_MIN);
}

TEST_F(GxfParameterSetInt64_Test, UpperBoundryValue) {
  GXF_ASSERT_SUCCESS(GxfParameterSetInt64(context, cid, "int64_var", INT64_MAX));
  GXF_ASSERT_SUCCESS(GxfParameterGetInt64(context, cid, "int64_var", &value));
  GXF_ASSERT_EQ(value, INT64_MAX);
}
