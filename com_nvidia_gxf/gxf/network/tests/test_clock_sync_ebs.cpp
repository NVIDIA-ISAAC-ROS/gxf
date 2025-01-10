/*
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gtest/gtest.h"

#include "gxf/core/expected.hpp"
#include "gxf/core/gxf.h"

namespace nvidia {
namespace gxf {
namespace test {

namespace {

constexpr const char* kExtensions[] = {
    "gxf/std/libgxf_std.so",
    "gxf/serialization/libgxf_serialization.so",
    "gxf/network/libgxf_network.so",
    "gxf/network/tests/libgxf_test_clock_sync_helpers_factory.so",
    "gxf/test/extensions/libgxf_test.so",
};

constexpr GxfLoadExtensionsInfo extension_info = { kExtensions, 5, nullptr, 0, nullptr };

}

TEST(TestClockSync, PrimaryToSecondaryEventBased) {
  gxf_context_t context1;
  ASSERT_EQ(GxfContextCreate(&context1), GXF_SUCCESS);

  gxf_context_t context2;
  ASSERT_EQ(GxfContextCreate(&context2), GXF_SUCCESS);

  ASSERT_EQ(GxfLoadExtensions(context1, &extension_info), GXF_SUCCESS);
  ASSERT_EQ(GxfLoadExtensions(context2, &extension_info), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphLoadFile(context1, "gxf/network/tests/test_clock_sync_primary_ebs.yaml"),
            GXF_SUCCESS);
  ASSERT_EQ(GxfGraphLoadFile(context2,
            "gxf/network/tests/test_clock_sync_secondary_ebs.yaml"), GXF_SUCCESS);

  ASSERT_EQ(GxfSetSeverity(context1, gxf_severity_t::GXF_SEVERITY_DEBUG), GXF_SUCCESS);
  ASSERT_EQ(GxfSetSeverity(context2, gxf_severity_t::GXF_SEVERITY_DEBUG), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphActivate(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphActivate(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphRunAsync(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphRunAsync(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphWait(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphWait(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphDeactivate(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphDeactivate(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfContextDestroy(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context2), GXF_SUCCESS);
}

}  // namespace test
}  // namespace gxf
}  // namespace nvidia
