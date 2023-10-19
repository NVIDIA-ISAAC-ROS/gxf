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
#include "gxf/core/handle.hpp"
#include "gxf/std/clock.hpp"

namespace {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

}  // namespace

TEST(SchedulingTerms, RealtimeClockEpoch) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));

  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, "gxf/test/unit/test_realtime_clock.yaml"));

  gxf_uid_t eid;
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "clock", &eid));
  gxf_tid_t tid;
  GXF_ASSERT_SUCCESS(
      GxfComponentTypeId(context, nvidia::TypenameAsString<nvidia::gxf::RealtimeClock>(), &tid));
  gxf_uid_t cid;
  GxfComponentFind(context, eid, tid, "default", nullptr, &cid);
  auto maybe_clock = nvidia::gxf::Handle<nvidia::gxf::RealtimeClock>::Create(context, cid);
  EXPECT_TRUE(maybe_clock);
  const nvidia::gxf::Handle<nvidia::gxf::RealtimeClock>& clock = maybe_clock.value();

  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));

  for (int i = 0; i < 10; i++) {
    const double clock_time = clock->time();
    const double wall_time = std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    EXPECT_NEAR(clock_time, wall_time, 1e-4);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  GXF_ASSERT_SUCCESS(GxfGraphInterrupt(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}
