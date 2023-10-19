/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <chrono>
#include <cstring>
#include <thread>
#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"

namespace {

constexpr const char* kStdExtensionFilename = "gxf/std/libgxf_std.so";
constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

}  // namespace

class GxfComponentAddToInterface_Test : public ::testing::Test {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
    GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::BlockMemoryPool", &tid));
    GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "test", &cid));
    GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, "gxf/core/tests/test_app_GxfComponentAddToInterface.yaml"));
  }

 protected:
 gxf_context_t context = kNullContext;
 const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
 gxf_uid_t eid = kNullUid;
 gxf_uid_t kNulleid;
 const GxfEntityCreateInfo entity_create_info = {0};
 gxf_tid_t tid = GxfTidNull();
 gxf_uid_t cid = kNullUid;
};

TEST_F(GxfComponentAddToInterface_Test, ValidParameter) {
  GXF_ASSERT_SUCCESS(GxfComponentAddToInterface(context, eid, cid, "Test_component"));
}

TEST_F(GxfComponentAddToInterface_Test, InvalidEntity) {
  GXF_ASSERT_EQ(GxfComponentAddToInterface(context, kNulleid, cid, "Test_component"),GXF_ENTITY_NOT_FOUND)
}

TEST_F(GxfComponentAddToInterface_Test, InvalidContext) {
  GXF_ASSERT_EQ(GxfComponentAddToInterface(kNullContext, eid, cid, "Test_component"),GXF_CONTEXT_INVALID);
}

TEST_F(GxfComponentAddToInterface_Test, ComponentAfterInitialization ) {
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "rx0", &eid));
  GXF_ASSERT_SUCCESS(GxfEntityDeactivate(context, eid));
  ASSERT_NE(eid,kNullUid);
  GXF_ASSERT_SUCCESS(GxfEntityActivate(context,eid));
  GXF_ASSERT_EQ(GxfComponentAddToInterface(context, eid, cid, "New_component"),GXF_ENTITY_CAN_NOT_ADD_COMPONENT_AFTER_INITIALIZATION);
  GXF_ASSERT_SUCCESS(GxfEntityDeactivate(context, eid));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}
