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

class GxfComponentFindAndGetPtr_Test : public ::testing::Test {
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
  gxf_uid_t eid1;
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  const GxfEntityCreateInfo entity_create_info = {0};
  gxf_tid_t tid = GxfTidNull();
  gxf_uid_t cid = kNullUid;
  int32_t offset;
  void* item_ptr = nullptr;
  void* comp_ptr = nullptr;
};

TEST_F(GxfComponentFindAndGetPtr_Test, NoComponent) {
  GXF_ASSERT_SUCCESS(GxfEntityGetItemPtr(context, eid, &item_ptr));
  GXF_ASSERT_EQ(
      GxfComponentFindAndGetPtr(context, eid, item_ptr, tid, "test1", &offset, &cid, &comp_ptr),
      GXF_ENTITY_COMPONENT_NOT_FOUND);
}

TEST_F(GxfComponentFindAndGetPtr_Test, InvalidContext) {
  GXF_ASSERT_SUCCESS(GxfEntityGetItemPtr(context, eid, &item_ptr));
  GXF_ASSERT_EQ(
      GxfComponentFindAndGetPtr(NULL, eid, item_ptr, tid, "test", &offset, &cid, &comp_ptr),
      GXF_CONTEXT_INVALID);
}

TEST_F(GxfComponentFindAndGetPtr_Test, NullItem) {
  GXF_ASSERT_EQ(
      GxfComponentFindAndGetPtr(context, eid, nullptr, tid, "test", &offset, &cid, &comp_ptr),
      GXF_ARGUMENT_NULL);
}

TEST_F(GxfComponentFindAndGetPtr_Test, NullCompPtr) {
  GXF_ASSERT_SUCCESS(GxfEntityGetItemPtr(context, eid, &item_ptr));
  GXF_ASSERT_EQ(
      GxfComponentFindAndGetPtr(context, eid, item_ptr, tid, "test", &offset, &cid, nullptr),
      GXF_ARGUMENT_NULL);
}

TEST_F(GxfComponentFindAndGetPtr_Test, CompPtrExists) {
  GXF_ASSERT_SUCCESS(GxfEntityGetItemPtr(context, eid, &item_ptr));
  GXF_ASSERT_SUCCESS(GxfComponentFindAndGetPtr(context, eid, item_ptr, tid, "testcomponentname",
                                               nullptr, &cid, &comp_ptr));
  GXF_ASSERT_EQ(
      GxfComponentFindAndGetPtr(context, eid, item_ptr, tid, "test", &offset, &cid, &comp_ptr),
      GXF_ARGUMENT_INVALID);
}
