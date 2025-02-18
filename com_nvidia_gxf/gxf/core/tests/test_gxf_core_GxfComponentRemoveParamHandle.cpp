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

class GxfParameterGetHandle_Test : public ::testing::Test {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
    GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));
    GxfComponentTypeId(context, "nvidia::gxf::GreedyScheduler", &tid);
    GxfComponentAdd(context, eid, tid, "scheduler", &cid);
    GxfComponentTypeId(context, "nvidia::gxf::RealtimeClock", &clock_tid);
    GxfComponentAdd(context, eid, clock_tid, "clock", &clock_cid);
    GXF_ASSERT_SUCCESS(GxfParameterSetHandle(context, cid, "clock", clock_cid));
  }

  void TearDown() {
    GXF_ASSERT_SUCCESS(GxfComponentRemoveWithUID(context, cid));
    GXF_ASSERT_SUCCESS(GxfComponentRemoveWithUID(context, clock_cid));
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
  gxf_tid_t clock_tid;
  gxf_uid_t clock_cid;
  gxf_uid_t handle_cid;
};

TEST_F(GxfParameterGetHandle_Test, ValidParameter) {
  GXF_ASSERT_SUCCESS(GxfParameterGetHandle(context, cid, "clock", &handle_cid));
  GXF_ASSERT_EQ(handle_cid, clock_cid);
}

TEST_F(GxfParameterGetHandle_Test, InvalidContext) {
  GXF_ASSERT_EQ(GxfParameterGetHandle(kNullContext, cid, "clock", &handle_cid), GXF_CONTEXT_INVALID);
}

TEST_F(GxfParameterGetHandle_Test, InvalidParameter) {
  GXF_ASSERT_EQ(GxfParameterGetHandle(context, kNullUid, "clock", &handle_cid),GXF_PARAMETER_NOT_FOUND);
}

TEST_F(GxfParameterGetHandle_Test, Nullpointer) {
  GXF_ASSERT_EQ(GxfParameterGetHandle(context, cid, "clock", kNullUid),GXF_ARGUMENT_NULL);
}

TEST_F(GxfParameterGetHandle_Test, InvalidType) {
  bool bool_var = true;
  GXF_ASSERT_SUCCESS(GxfParameterSetBool(context, cid, "bool_var", bool_var));
  GXF_ASSERT_EQ(GxfParameterGetHandle(context, cid, "bool_var", &handle_cid),GXF_PARAMETER_INVALID_TYPE);
}
