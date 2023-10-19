/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/scheduling_terms.hpp"

#include "gtest/gtest.h"

namespace nvidia {
namespace gxf {

class TestBooleanSchedulingTerm : public testing::TestWithParam<bool> {};

TEST_P(TestBooleanSchedulingTerm, createEoSMessage) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  constexpr const char* kExtensions[] = {
      "gxf/std/libgxf_std.so",
  };
  const GxfLoadExtensionsInfo info{kExtensions, 1, nullptr, 0, nullptr};

  ASSERT_EQ(GxfLoadExtensions(context, &info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::BooleanSchedulingTerm", &tid), GXF_SUCCESS);
  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "term", &cid), GXF_SUCCESS);

  const bool tick_initial_condition = GetParam();

  ASSERT_EQ(GxfParameterSetBool(context, cid, "enable_tick", tick_initial_condition), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);
  BooleanSchedulingTerm* term = static_cast<BooleanSchedulingTerm*>(pointer);

  ASSERT_EQ(term->initialize(), GXF_SUCCESS);

  // Default shall keep tick enabled.
  ASSERT_EQ(term->checkTickEnabled(), tick_initial_condition);

  ASSERT_TRUE(term->enable_tick());
  ASSERT_EQ(term->checkTickEnabled(), true);

  constexpr int64_t DURATION = 123456;
  SchedulingConditionType type;
  int64_t duration = DURATION;

  ASSERT_EQ(term->onExecute_abi(0), GXF_SUCCESS);
  ASSERT_EQ(term->onExecute_abi(100), GXF_SUCCESS);

  int64_t now = 123456;
  ASSERT_EQ(term->check_abi(123456, &type, &now), GXF_SUCCESS);
  ASSERT_EQ(type, SchedulingConditionType::READY);
  ASSERT_EQ(duration, DURATION);

  ASSERT_TRUE(term->disable_tick());
  ASSERT_EQ(term->checkTickEnabled(), false);

  ASSERT_EQ(term->check_abi(123456, &type, &now), GXF_SUCCESS);
  ASSERT_EQ(type, SchedulingConditionType::NEVER);
  ASSERT_EQ(duration, DURATION);

  ASSERT_TRUE(term->enable_tick());
  ASSERT_EQ(term->checkTickEnabled(), true);

  ASSERT_EQ(term->check_abi(123456, &type, &now), GXF_SUCCESS);
  ASSERT_EQ(type, SchedulingConditionType::READY);
  ASSERT_EQ(duration, DURATION);

  ASSERT_EQ(term->deinitialize(), GXF_SUCCESS);

  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

const bool tick_initial_conditions[] = {true, false};

INSTANTIATE_TEST_CASE_P(BooleanSchedulingTerm,
  TestBooleanSchedulingTerm,
  testing::ValuesIn(tick_initial_conditions));

}  // namespace gxf
}  // namespace nvidia
