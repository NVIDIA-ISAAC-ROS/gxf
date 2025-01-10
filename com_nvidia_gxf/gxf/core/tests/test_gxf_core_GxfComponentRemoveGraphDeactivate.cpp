/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <string.h>
#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"
#include "gxf/core/handle.hpp"
#include "gxf/std/clock.hpp"

namespace {
constexpr const char* kExtensions[] = {"gxf/std/libgxf_std.so",
                                       "gxf/test/extensions/libgxf_test.so"};
}  // namespace

class GxfGraphDeactivateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info_std));
  }

  gxf_context_t context = kNullContext;
  const GxfLoadExtensionsInfo info_std{kExtensions, 2, nullptr, 0, nullptr};
};

TEST_F(GxfGraphDeactivateTest, DeactivateBeforeActivate) {
  gxf_uid_t eid = kNullUid;
  const GxfEntityCreateInfo entity_create_info = {"abc", 1};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);
  GXF_ASSERT_NE(eid, kNullUid);
  gxf_tid_t tid = GxfTidNull();
  GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::GreedyScheduler", &tid));
  gxf_uid_t cid = kNullUid;
  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "scheduler", &cid));

  gxf_tid_t tid_clock;
  GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::RealtimeClock", &tid_clock));
  gxf_uid_t cid_clock;
  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid_clock, "clock", &cid_clock));
  GXF_ASSERT_SUCCESS(GxfComponentRemove(context, eid, tid, "scheduler"));
  GXF_ASSERT_SUCCESS(GxfComponentRemove(context, eid, tid_clock, "clock"));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
}

TEST_F(GxfGraphDeactivateTest, DeactivateAfterActivate) {
  gxf_uid_t eid = kNullUid;
  const GxfEntityCreateInfo entity_create_info = {"abc", 1};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);
  GXF_ASSERT_NE(eid, kNullUid);
  gxf_tid_t tid = GxfTidNull();
  GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::DoubleBufferTransmitter", &tid));
  gxf_uid_t cid = kNullUid;
  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "c", &cid));
  GXF_ASSERT_SUCCESS(GxfComponentRemove(context, eid, tid, "c"));
  GXF_ASSERT_EQ(GxfComponentRemoveWithUID(context, cid), GXF_ENTITY_NOT_FOUND);
  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "c", &cid));
  GXF_ASSERT_NE(cid, kNullUid);
  GXF_LOG_INFO("ACTIVATING GRAPH");
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_LOG_INFO("ACTIVATED GRAPH");
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfComponentRemove(context, eid, tid, "c"));
}

TEST_F(GxfGraphDeactivateTest, DeactivateAfterRun) {
  gxf_uid_t eid = kNullUid;
  const GxfEntityCreateInfo entity_create_info = {"abc", 1};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);
  GXF_ASSERT_NE(eid, kNullUid);
  gxf_tid_t tid = GxfTidNull();
  GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::GreedyScheduler", &tid));
  gxf_uid_t cid = kNullUid;
  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "scheduler", &cid));

  gxf_tid_t tid_clock;
  GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::RealtimeClock", &tid_clock));
  gxf_uid_t cid_clock;
  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid_clock, "clock", &cid_clock));
  GXF_ASSERT_SUCCESS(GxfParameterSetHandle(context, cid, "clock", cid_clock));
  GXF_ASSERT_NE(cid, kNullUid);
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRun(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfComponentRemove(context, eid, tid, "scheduler"));
}

TEST_F(GxfGraphDeactivateTest, ValidDeactivateAfterRunAsync) {
  gxf_uid_t eid = kNullUid;
  const GxfEntityCreateInfo entity_create_info = {"abc", 1};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);
  GXF_ASSERT_NE(eid, kNullUid);
  gxf_tid_t tid = GxfTidNull();
  GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::GreedyScheduler", &tid));
  gxf_uid_t cid = kNullUid;
  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "scheduler", &cid));

  gxf_tid_t tid_clock;
  GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::RealtimeClock", &tid_clock));
  gxf_uid_t cid_clock;
  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid_clock, "clock", &cid_clock));
  GXF_ASSERT_SUCCESS(GxfParameterSetHandle(context, cid, "clock", cid_clock));
  GXF_ASSERT_NE(cid, kNullUid);
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfComponentRemoveWithUID(context, cid));
}