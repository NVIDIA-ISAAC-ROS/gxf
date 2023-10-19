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

class GxfParameterGetFloat64_Test : public ::testing::Test {
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
  double value;
};

TEST_F(GxfParameterGetFloat64_Test, ValidParameter) {
  double float64_var = 101.1;
  GXF_ASSERT_SUCCESS(GxfParameterSetFloat64(context, cid, "float64_var", float64_var));

  GXF_ASSERT_SUCCESS(GxfParameterGetFloat64(context, cid, "float64_var", &value));
  GXF_ASSERT_EQ(value, float64_var);
}

TEST_F(GxfParameterGetFloat64_Test, InvalidContext) {
  GXF_ASSERT_EQ(GxfParameterGetFloat64(kNullContext, cid, "float64_var", &value),
                GXF_CONTEXT_INVALID);
}

TEST_F(GxfParameterGetFloat64_Test, InvalidUid) {
  GXF_ASSERT_EQ(GxfParameterGetFloat64(context, kNullUid, "float64_var", &value),
                GXF_PARAMETER_NOT_FOUND);
}

TEST_F(GxfParameterGetFloat64_Test, InvalidKey) {
  double float64_var = 101.1;
  GXF_ASSERT_SUCCESS(GxfParameterSetFloat64(context, cid, "float64_var", float64_var));
  GXF_ASSERT_EQ(GxfParameterGetFloat64(context, cid, "float64_invalid_var", &value),
                GXF_PARAMETER_NOT_FOUND);
}

TEST_F(GxfParameterGetFloat64_Test, InvalidParameterType) {
  int64_t int64_var = 101;
  GXF_ASSERT_SUCCESS(GxfParameterSetInt64(context, cid, "int64_var", int64_var));
  GXF_ASSERT_EQ(GxfParameterGetFloat64(context, cid, "int64_var", &value),
                GXF_PARAMETER_INVALID_TYPE);
}
