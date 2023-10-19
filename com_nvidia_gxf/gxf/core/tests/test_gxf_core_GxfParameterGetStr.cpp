/*
Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.

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

class GxfParameterGetStr_Test : public ::testing::Test {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
    GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::BlockMemoryPool", &tid));
    GxfComponentAdd(context, eid, tid, "capacity", &cid);
    GxfParameterSetStr(context,cid,"test","Test Message");
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
  char var1 = 'g';
  const char *var2 = &var1;
  const char ** value= &var2;
};

TEST_F(GxfParameterGetStr_Test, ValidParameter) {
  GXF_ASSERT_SUCCESS(GxfParameterGetStr(context,cid,"test",value));
}

TEST_F(GxfParameterGetStr_Test, ParameterNotFound) {
  GXF_ASSERT_EQ((GxfParameterGetStr(context,cid,"capacity",value)),GXF_PARAMETER_NOT_FOUND);
}

TEST_F(GxfParameterGetStr_Test, NullArgument) {
  GXF_ASSERT_EQ((GxfParameterGetStr(context,cid,"capacity",nullptr)),GXF_ARGUMENT_NULL);
}

TEST_F(GxfParameterGetStr_Test, InvalidParameter) {
  GXF_ASSERT_EQ((GxfParameterGetStr(kNullContext,cid,"capacity",value)),GXF_CONTEXT_INVALID);
}
