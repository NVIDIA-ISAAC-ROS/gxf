/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <cstring>

#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"

namespace {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

}  // namespace

class GxfParameterSetStr_Test : public ::testing::Test {
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
  const char* value;
};

TEST_F(GxfParameterSetStr_Test, ValidParameter) {
  const char* str_var = "foo";
  GXF_ASSERT_SUCCESS(GxfParameterSetStr(context, cid, "str_var", str_var));

  GXF_ASSERT_SUCCESS(GxfParameterGetStr(context, cid, "str_var", &value));
  GXF_ASSERT_STREQ(value, str_var);
}

TEST_F(GxfParameterSetStr_Test, InvalidContext) {
  const char* str_var = "foo";
  GXF_ASSERT_EQ(GxfParameterSetStr(kNullContext, cid, "str_var", str_var),
                GXF_CONTEXT_INVALID);
}

TEST_F(GxfParameterSetStr_Test, OverwritingValidParameterType) {
  const char* str_var_1 = "foo";
  const char* str_var_2 = "foo1";
  GXF_ASSERT_SUCCESS(GxfParameterSetStr(context, cid, "str_var", str_var_1));
  GXF_ASSERT_SUCCESS(GxfParameterGetStr(context, cid, "str_var", &value));
  GXF_ASSERT_STREQ(value, str_var_1);
  GXF_ASSERT_SUCCESS(GxfParameterSetStr(context, cid, "str_var", str_var_2));
  GXF_ASSERT_SUCCESS(GxfParameterGetStr(context, cid, "str_var", &value));
  GXF_ASSERT_STREQ(value, str_var_2);
}

TEST_F(GxfParameterSetStr_Test, SettingInvalidParameterType) {
  int64_t int64_var = 129;
  const char* str_var = "foo";
  GXF_ASSERT_SUCCESS(GxfParameterSetInt64(context, cid, "int64_var", int64_var));
  GXF_ASSERT_EQ(GxfParameterSetStr(context, cid, "int64_var", str_var),
                GXF_PARAMETER_INVALID_TYPE);
}
