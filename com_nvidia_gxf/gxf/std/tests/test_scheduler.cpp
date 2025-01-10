/*
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gtest/gtest.h"
#include "common/assert.hpp"
#include "gxf/core/gxf.h"

namespace nvidia {
namespace gxf {

constexpr const char* kExtensions[] = {
    "gxf/std/libgxf_std.so",
    "gxf/test/extensions/libgxf_test.so",
};

TEST(SchedulerExit, Greedy) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfLoadExtensionsInfo info{kExtensions, 2, nullptr, 0, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, "gxf/std/tests/apps/test_scheduler_exit.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, "gxf/std/tests/apps/test_greedy_sched_param.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  ASSERT_EQ(GxfGraphRun(context), GXF_FAILURE);
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(SchedulerExit, GreedyRunAsync) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfLoadExtensionsInfo info{kExtensions, 2, nullptr, 0, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, "gxf/std/tests/apps/test_scheduler_exit.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, "gxf/std/tests/apps/test_greedy_sched_param.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_EQ(GxfGraphWait(context), GXF_FAILURE);
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(SchedulerExit, Multithread) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfLoadExtensionsInfo info{kExtensions, 2, nullptr, 0, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, "gxf/std/tests/apps/test_scheduler_exit.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, "gxf/std/tests/apps/test_multi_sched_param.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  ASSERT_EQ(GxfGraphRun(context), GXF_FAILURE);
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(SchedulerExit, MultithreadRunAsync) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfLoadExtensionsInfo info{kExtensions, 2, nullptr, 0, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, "gxf/std/tests/apps/test_scheduler_exit.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, "gxf/std/tests/apps/test_multi_sched_param.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_EQ(GxfGraphWait(context), GXF_FAILURE);
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(SchedulerExit, EventBasedScheduler) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfLoadExtensionsInfo info{kExtensions, 2, nullptr, 0, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, "gxf/std/tests/apps/test_scheduler_exit.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, "gxf/std/tests/apps/test_ebs_param.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  ASSERT_EQ(GxfGraphRun(context), GXF_FAILURE);
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(SchedulerExit, EventBasedSchedulerRunAsync) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfLoadExtensionsInfo info{kExtensions, 2, nullptr, 0, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, "gxf/std/tests/apps/test_scheduler_exit.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, "gxf/std/tests/apps/test_ebs_param.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_EQ(GxfGraphWait(context), GXF_FAILURE);
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

}  // namespace gxf
}  // namespace nvidia
