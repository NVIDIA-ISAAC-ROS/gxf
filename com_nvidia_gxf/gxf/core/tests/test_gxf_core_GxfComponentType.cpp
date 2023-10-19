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

class GxfComponentType_Test : public ::testing::Test {
 public:
  void SetUp() {
      GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
      GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
      GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));
      GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::BlockMemoryPool", &tid));
      GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "testcomponentname", &cid));
  }

  void TearDown() {
    GXF_ASSERT_SUCCESS(GxfEntityDestroy(context, eid));
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
  }

 protected:
    gxf_context_t context = kNullContext;
    gxf_uid_t eid = kNullUid;
    const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
    const GxfEntityCreateInfo entity_create_info = {0};
    gxf_tid_t tid = GxfTidNull();
    gxf_uid_t cid = kNullUid;
    gxf_uid_t cid1 = kNullUid;
};

TEST_F(GxfComponentType_Test, ValidParameter) {
  GXF_ASSERT_SUCCESS(GxfComponentType(context, cid, &tid));
}

TEST_F(GxfComponentType_Test, NoComponent) {
  GXF_ASSERT_EQ(GxfComponentType(context, cid1, &tid),GXF_ENTITY_COMPONENT_NOT_FOUND);
}

TEST_F(GxfComponentType_Test, InvalidContext) {
  GXF_ASSERT_EQ(GxfComponentType(NULL, cid, &tid),GXF_CONTEXT_INVALID);
}