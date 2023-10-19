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

class GxfParameterSetUInt64_Test : public ::testing::Test {
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
  uint64_t value;
};

TEST_F(GxfParameterSetUInt64_Test, ValidParameter) {
  uint64_t uint64_var = 184;
  GXF_ASSERT_SUCCESS(GxfParameterSetUInt64(context, cid, "uint64_var", uint64_var));
  GXF_ASSERT_SUCCESS(GxfParameterGetUInt64(context, cid, "uint64_var", &value));
  GXF_ASSERT_EQ(value, uint64_var);
}

TEST_F(GxfParameterSetUInt64_Test, InvalidContext) {
  uint64_t uint64_var = 184;
  GXF_ASSERT_EQ(GxfParameterSetUInt64(kNullContext, cid, "uint64_var", uint64_var),
                GXF_CONTEXT_INVALID);
}

TEST_F(GxfParameterSetUInt64_Test, OverwritingValidParameterType) {
  uint64_t uint64_var_1 = 184;
  uint64_t uint64_var_2 = 121;
  GXF_ASSERT_SUCCESS(GxfParameterSetUInt64(context, cid, "uint64_var", uint64_var_1));
  GXF_ASSERT_SUCCESS(GxfParameterGetUInt64(context, cid, "uint64_var", &value));
  GXF_ASSERT_EQ(value, uint64_var_1);
  GXF_ASSERT_SUCCESS(GxfParameterSetUInt64(context, cid, "uint64_var", uint64_var_2));
  GXF_ASSERT_SUCCESS(GxfParameterGetUInt64(context, cid, "uint64_var", &value));
  GXF_ASSERT_EQ(value, uint64_var_2);
}

TEST_F(GxfParameterSetUInt64_Test, SettingInvalidParameterType) {
  uint64_t uint64_var = 129;
  double float64_var = 101.1;
  GXF_ASSERT_SUCCESS(GxfParameterSetFloat64(context, cid, "float64_var", float64_var));
  GXF_ASSERT_EQ(GxfParameterSetUInt64(context, cid, "float64_var", uint64_var),
                GXF_PARAMETER_INVALID_TYPE);
}

TEST_F(GxfParameterSetUInt64_Test, LowerBoundryValue) {
  GXF_ASSERT_SUCCESS(GxfParameterSetUInt64(context, cid, "uint64_var", 0));
  GXF_ASSERT_SUCCESS(GxfParameterGetUInt64(context, cid, "uint64_var", &value));
  GXF_ASSERT_EQ(value, 0);
}

TEST_F(GxfParameterSetUInt64_Test, UpperBoundryValue) {
  GXF_ASSERT_SUCCESS(GxfParameterSetUInt64(context, cid, "uint64_var", UINT64_MAX));
  GXF_ASSERT_SUCCESS(GxfParameterGetUInt64(context, cid, "uint64_var", &value));
  GXF_ASSERT_EQ(value, UINT64_MAX);
}
