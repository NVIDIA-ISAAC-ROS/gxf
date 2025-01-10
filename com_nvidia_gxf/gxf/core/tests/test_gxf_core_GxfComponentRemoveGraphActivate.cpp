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
#include "gxf/core/gxf.h"

#include "gtest/gtest.h"

namespace {
constexpr const char* kExtensions[] = {"gxf/std/libgxf_std.so",
                                       "gxf/test/extensions/libgxf_test.so"};
}  // namespace

class GxfGraphActivateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info_std));
  }

  gxf_context_t context = kNullContext;
  const GxfLoadExtensionsInfo info_std{kExtensions, 2, nullptr, 0, nullptr};
};

TEST_F(GxfGraphActivateTest, Valid1024EntitiesWith1024ComponentsEach) {
  for (int i = 0; i < 1024; i++) {
    gxf_uid_t eid = kNullUid;
    std::string entity_name = "e" + std::to_string(i);
    const GxfEntityCreateInfo entity_create_info = {
        entity_name.c_str(), GxfEntityCreateFlagBits::GXF_ENTITY_CREATE_PROGRAM_BIT};

    ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

    for (int j = 0; j < 1024; j++) {
      gxf_tid_t tid = GxfTidNull();
      GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::DoubleBufferTransmitter", &tid));
      gxf_uid_t cid = kNullUid;
      GXF_ASSERT_SUCCESS(
          GxfComponentAdd(context, eid, tid, ("c" + std::to_string(j)).c_str(), &cid));
      GXF_ASSERT_NE(cid, kNullUid);
    }
  }
  GXF_ASSERT_EQ(GxfGraphActivate(context), GXF_SUCCESS);
  gxf_uid_t eid = kNullUid;
  gxf_tid_t tid = GxfTidNull();
  GxfEntityFind(context, "e1023", &eid);
  GxfComponentTypeId(context, "nvidia::gxf::DoubleBufferTransmitter", &tid);
  GXF_ASSERT_EQ(GxfComponentRemove(context, eid, tid, "c1023" ), GXF_ENTITY_CAN_NOT_REMOVE_COMPONENT_AFTER_INITIALIZATION);
}
