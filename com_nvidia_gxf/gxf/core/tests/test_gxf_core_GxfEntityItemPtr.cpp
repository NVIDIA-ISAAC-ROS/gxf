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
#include "gxf/core/entity_item.hpp"
#include "gxf/core/gxf.h"

namespace {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

}  // namespace

class GxfEntityItemPtr_Test : public ::testing::Test {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
    GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));
    item_ptr = nullptr;
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
  gxf_uid_t eid1;
  void* item_ptr;
};

TEST_F(GxfEntityItemPtr_Test, ValidParameter) {
  GXF_ASSERT_SUCCESS(GxfEntityGetItemPtr(context, eid, &item_ptr));
  GXF_ASSERT_EQ((reinterpret_cast<nvidia::gxf::EntityItem*>(item_ptr))->uid, eid);
}

TEST_F(GxfEntityItemPtr_Test, NullContext) {
  GXF_ASSERT_EQ(GxfEntityGetItemPtr(kNullContext, eid, &item_ptr), GXF_CONTEXT_INVALID);
}

TEST_F(GxfEntityItemPtr_Test, InvalidEid) {
  GXF_ASSERT_EQ(GxfEntityGetItemPtr(context, eid1, &item_ptr), GXF_ENTITY_NOT_FOUND);
}

TEST_F(GxfEntityItemPtr_Test, NullPtr) {
  GXF_ASSERT_EQ(GxfEntityGetItemPtr(context, eid1, nullptr), GXF_ARGUMENT_NULL);
}

TEST_F(GxfEntityItemPtr_Test, InvalidPtr) {
  GXF_ASSERT_SUCCESS(GxfEntityGetItemPtr(context, eid, &item_ptr));
  GXF_ASSERT_EQ(GxfEntityGetItemPtr(context, eid, &item_ptr), GXF_ARGUMENT_INVALID);
}
