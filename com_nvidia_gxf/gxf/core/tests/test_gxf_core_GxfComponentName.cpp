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

class GxfComponentName_Test : public ::testing::Test {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
    GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::BlockMemoryPool", &tid));
    GxfComponentAdd(context, eid, tid, "capacity", &uid);
  }

  void TearDown() {
    GXF_ASSERT_SUCCESS(GxfEntityDestroy(context, eid));
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
  }

 protected:
  gxf_context_t context = kNullContext;
  gxf_context_t null_context = kNullContext;
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  gxf_uid_t eid = kNullUid;
  const GxfEntityCreateInfo entity_create_info = {0};
  gxf_tid_t tid = GxfTidNull();
  gxf_uid_t uid = kNullUid;
  gxf_uid_t cid = kNullUid;
  const char *var1 = "testcomponentname";
  const char ** value= &var1;
  const char** null_value = nullptr;
};

TEST_F(GxfComponentName_Test, NullArgument) {
  GXF_ASSERT_EQ((GxfComponentName(context, uid, null_value)),GXF_ARGUMENT_NULL);
}

TEST_F(GxfComponentName_Test, ValidParameter) {
  GXF_ASSERT_SUCCESS(GxfComponentName(context, uid, value));
}

TEST_F(GxfComponentName_Test, InvalidContext) {
  GXF_ASSERT_EQ((GxfComponentName(null_context, uid, value)),GXF_CONTEXT_INVALID);
}

TEST_F(GxfComponentName_Test, ParameterNotFound) {
  GXF_ASSERT_EQ((GxfComponentName(context, cid, value)),GXF_PARAMETER_NOT_FOUND);
}