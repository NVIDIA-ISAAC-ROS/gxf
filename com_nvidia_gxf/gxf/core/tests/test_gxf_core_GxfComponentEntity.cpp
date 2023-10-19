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

class GxfComponentEntity_Test : public ::testing::Test {
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
    gxf_uid_t eid = kNullUid;
    const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
    const GxfEntityCreateInfo entity_create_info = {0};
    gxf_tid_t tid = GxfTidNull();
    gxf_uid_t cid = kNullUid;
    gxf_uid_t eid1;
};

TEST_F(GxfComponentEntity_Test, ValidParameter) {
  GXF_ASSERT_SUCCESS(GxfComponentEntity(context, cid, &eid1));
}

TEST_F(GxfComponentEntity_Test, InvalidContext) {
  GXF_ASSERT_EQ((GxfComponentEntity(NULL, cid, &eid1)),GXF_CONTEXT_INVALID);
}

TEST_F(GxfComponentEntity_Test, InvalidEntity) {
  GXF_ASSERT_EQ(GxfComponentEntity(context, eid, &eid1), GXF_ENTITY_NOT_FOUND);
}