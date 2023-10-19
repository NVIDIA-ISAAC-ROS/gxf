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

class GxfComponentAdd_Test : public ::testing::Test {
 public:
  void SetUp() {
      GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
      GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
      GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));
      GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::BlockMemoryPool", &tid));
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

TEST_F(GxfComponentAdd_Test, ValidParameter) {
  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "test", &cid));
}

TEST_F(GxfComponentAdd_Test, InvalidEntity) {
  GXF_ASSERT_EQ(GxfComponentAdd(context, eid1, tid, "test", &cid), GXF_ENTITY_NOT_FOUND);
}

TEST_F(GxfComponentAdd_Test, InvalidTid) {
  gxf_tid_t tid1 = GxfTidNull();
  GXF_ASSERT_EQ(GxfComponentAdd(context, eid, tid1, "test", &cid), GXF_FACTORY_UNKNOWN_TID);
}

TEST_F(GxfComponentAdd_Test, InvalidContext) {
  GXF_ASSERT_EQ(GxfComponentAdd(NULL, eid, tid, "test", &cid),GXF_CONTEXT_INVALID);
}

TEST_F(GxfComponentAdd_Test, MaxComponents) {
  for (auto i = 0; i < kMaxComponents; ++i) {
    GXF_ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test", &cid),GXF_SUCCESS);
  }
  GXF_ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test", &cid), GXF_OUT_OF_MEMORY);
}
