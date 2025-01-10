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


class GxfComponentIsBase_Test : public ::testing::Test {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::PingTx", &derived));
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::Codelet", &base));
  }

  void TearDown() {
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
  }

 protected:
  gxf_context_t context = kNullContext;
  gxf_tid_t derived = GxfTidNull();
  gxf_tid_t base = GxfTidNull();
  gxf_tid_t null_tid = GxfTidNull();
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
};

TEST_F(GxfComponentIsBase_Test,InvalidTid){
    bool result {false};
    GXF_ASSERT_EQ((GxfComponentIsBase(context, null_tid, base, &result)),GXF_QUERY_NOT_FOUND);
}

TEST_F(GxfComponentIsBase_Test,ValidBase){
    bool result {false};
    GXF_ASSERT_SUCCESS(GxfComponentIsBase(context, derived, base, &result));
    GXF_ASSERT_TRUE(result);
}

TEST_F(GxfComponentIsBase_Test,InvalidBase){
    bool result {false};
    GXF_ASSERT_SUCCESS(GxfComponentIsBase(context, base, derived, &result));
    GXF_ASSERT_FALSE(result);
}
