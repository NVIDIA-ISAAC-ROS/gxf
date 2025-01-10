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

class GxfParameterGetBool_Test : public ::testing::Test {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
    GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::BlockMemoryPool", &tid));
    GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "test", &cid));
    GXF_ASSERT_SUCCESS(GxfParameterSetBool(context, cid, "bool_var", bool_var));
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
  gxf_uid_t cid1 = kNullUid;
  bool value;
  bool bool_var = false;
  bool* kNullval = nullptr;
};

TEST_F(GxfParameterGetBool_Test, ValidParameter) {
    GXF_ASSERT_SUCCESS(GxfParameterGetBool(context, cid, "bool_var", &value));
    GXF_ASSERT_EQ(value, bool_var);
}

TEST_F(GxfParameterGetBool_Test, InvalidParameter) {
    GXF_ASSERT_EQ(GxfParameterGetBool(context, cid, "bool_var",kNullval),GXF_ARGUMENT_NULL);
}

TEST_F(GxfParameterGetBool_Test, NoParameter) {
    GXF_ASSERT_EQ(GxfParameterGetBool(context, cid1, "bool_var",&value),GXF_PARAMETER_NOT_FOUND)
}

TEST_F(GxfParameterGetBool_Test, InvalidContext) {
  GXF_ASSERT_EQ(GxfParameterGetBool(kNullContext, cid, "bool_var",&value),GXF_CONTEXT_INVALID);
}

TEST_F(GxfParameterGetBool_Test, OverwritingValidParameterType) {
  bool bool_var_1 = "foo";
  bool bool_var_2 = "foo1";
  GXF_ASSERT_SUCCESS(GxfParameterSetBool(context, cid, "bool_var", bool_var_1));
  GXF_ASSERT_SUCCESS(GxfParameterGetBool(context, cid, "bool_var", &value));
  GXF_ASSERT_EQ(value, bool_var_1);
  GXF_ASSERT_SUCCESS(GxfParameterSetBool(context, cid, "bool_var", bool_var_2));
  GXF_ASSERT_SUCCESS(GxfParameterGetBool(context, cid, "bool_var", &value));
  GXF_ASSERT_EQ(value, bool_var_2);
}

TEST_F(GxfParameterGetBool_Test, OverwritingInvalidParameterType) {
  int64_t int64_var = 129;
  bool bool_var = true;
  GXF_ASSERT_SUCCESS(GxfParameterSetInt64(context, cid, "int64_var", int64_var));
  GXF_ASSERT_EQ(GxfParameterSetBool(context, cid, "int64_var", bool_var),GXF_PARAMETER_INVALID_TYPE);
}
