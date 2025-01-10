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

class GxfComponentAddAndGetPtr_Test : public ::testing::Test {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
    GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::BlockMemoryPool", &tid));
    item_ptr = nullptr;
    comp_ptr = nullptr;
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
  void* item_ptr;
  void* comp_ptr;
};

TEST_F(GxfComponentAddAndGetPtr_Test, ValidParameter) {
  GXF_ASSERT_SUCCESS(GxfEntityGetItemPtr(context, eid, &item_ptr));
  GXF_ASSERT_SUCCESS(GxfComponentAddAndGetPtr(context, item_ptr, tid, "test", &cid, &comp_ptr));
}

TEST_F(GxfComponentAddAndGetPtr_Test, InvalidEntity) {
  comp_ptr = nullptr;
  GXF_ASSERT_EQ(GxfComponentAddAndGetPtr(context, nullptr, tid, "test", &cid, &comp_ptr),
                GXF_ARGUMENT_NULL);
}

TEST_F(GxfComponentAddAndGetPtr_Test, NullCompPtr) {
  GXF_ASSERT_SUCCESS(GxfEntityGetItemPtr(context, eid, &item_ptr));
  GXF_ASSERT_EQ(GxfComponentAddAndGetPtr(context, item_ptr, tid, "test", &cid, nullptr),
                GXF_ARGUMENT_NULL);
}

TEST_F(GxfComponentAddAndGetPtr_Test, ExistingCompPtr) {
  GXF_ASSERT_SUCCESS(GxfEntityGetItemPtr(context, eid, &item_ptr));
  GXF_ASSERT_SUCCESS(GxfComponentAddAndGetPtr(context, item_ptr, tid, "test", &cid, &comp_ptr));
  GXF_ASSERT_EQ(GxfComponentAddAndGetPtr(context, item_ptr, tid, "test", &cid, &comp_ptr),
                GXF_ARGUMENT_INVALID);
}

TEST_F(GxfComponentAddAndGetPtr_Test, InvalidTid) {
  gxf_tid_t tid1 = GxfTidNull();
  GXF_ASSERT_SUCCESS(GxfEntityGetItemPtr(context, eid, &item_ptr));
  GXF_ASSERT_EQ(GxfComponentAddAndGetPtr(context, item_ptr, tid1, "test", &cid, &comp_ptr),
                GXF_FACTORY_UNKNOWN_TID);
}

TEST_F(GxfComponentAddAndGetPtr_Test, InvalidContext) {
  GXF_ASSERT_EQ(GxfComponentAddAndGetPtr(NULL, item_ptr, tid, "test", &cid, &comp_ptr),
                GXF_CONTEXT_INVALID);
}

TEST_F(GxfComponentAddAndGetPtr_Test, MaxComponents) {
  GXF_ASSERT_SUCCESS(GxfEntityGetItemPtr(context, eid, &item_ptr));
  for (auto i = 0; i < kMaxComponents; ++i) {
    GXF_ASSERT_EQ(GxfComponentAddAndGetPtr(context, item_ptr, tid, "test", &cid, &comp_ptr),
                  GXF_SUCCESS);
    comp_ptr = nullptr;
  }
  GXF_ASSERT_EQ(GxfComponentAddAndGetPtr(context, item_ptr, tid, "test", &cid, &comp_ptr),
                GXF_ENTITY_MAX_COMPONENTS_LIMIT_EXCEEDED);
}
