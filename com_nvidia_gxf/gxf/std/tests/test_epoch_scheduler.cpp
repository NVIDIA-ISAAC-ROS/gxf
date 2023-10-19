/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

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
#include "gxf/std/epoch_scheduler.hpp"

namespace {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";
constexpr const char* kExtensions[] = {
    "gxf/std/libgxf_std.so",
    "gxf/sample/libgxf_sample.so",
    "gxf/test/extensions/libgxf_test.so",
};

}  // namespace

TEST(EpochSchedulerTests, Step) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));

  const GxfLoadExtensionsInfo load_extension_info{kExtensions, 3, nullptr, 0,
                                                  nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &load_extension_info));
  GXF_ASSERT_SUCCESS(
      GxfGraphLoadFile(context, "gxf/std/tests/test_epoch_scheduler_app.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));

  std::this_thread::sleep_for(std::chrono::milliseconds(5));

  gxf_uid_t eid = 0;
  gxf_uid_t cid = 0;
  int32_t offset = 0;
  gxf_tid_t tid{0, 0};
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "scheduler", &eid));
  GXF_ASSERT_SUCCESS(
      GxfComponentFind(context, eid, tid, "epoch", &offset, &cid));

  nvidia::gxf::EpochScheduler* scheduler_ptr = nullptr;
  GXF_ASSERT_SUCCESS(GxfComponentPointer(
      context, cid, tid, reinterpret_cast<void**>(&scheduler_ptr)));

  EXPECT_NE(scheduler_ptr, nullptr);

  for (int i = 0; i < 12; ++i) {  // 1 for start(), one for rx, 10 for tx
    auto result = scheduler_ptr->runEpoch(0.1f);
    EXPECT_TRUE(result);
  }

  GXF_ASSERT_SUCCESS(GxfGraphInterrupt(context));

  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(EpochSchedulerTests, TimeBudget) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));

  const GxfLoadExtensionsInfo load_extension_info{kExtensions, 3, nullptr, 0,
                                                  nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &load_extension_info));
  GXF_ASSERT_SUCCESS(
      GxfGraphLoadFile(context, "gxf/std/tests/test_epoch_scheduler_app.yaml"));

  gxf_uid_t eid = 0;
  gxf_uid_t cid = 0;
  int32_t offset = 0;
  gxf_tid_t tid{0, 0};
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "scheduler", &eid));
  GXF_ASSERT_SUCCESS(
      GxfComponentFind(context, eid, tid, "epoch", &offset, &cid));

  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));

  std::this_thread::sleep_for(std::chrono::milliseconds(5));

  nvidia::gxf::EpochScheduler* scheduler_ptr = nullptr;
  GXF_ASSERT_SUCCESS(GxfComponentPointer(
      context, cid, tid, reinterpret_cast<void**>(&scheduler_ptr)));

  EXPECT_NE(scheduler_ptr, nullptr);

  auto result = scheduler_ptr->runEpoch(20.0);
  EXPECT_TRUE(result);
  GXF_ASSERT_SUCCESS(GxfGraphInterrupt(context));

  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}
