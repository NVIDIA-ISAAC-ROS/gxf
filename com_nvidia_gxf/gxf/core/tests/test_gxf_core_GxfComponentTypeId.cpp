/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

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


class GxfComponentTypeId_Test : public ::testing::Test {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  }

  void TearDown() {
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
  }

 protected:
  gxf_context_t context = kNullContext;
  gxf_tid_t tid = GxfTidNull();
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
};

TEST_F(GxfComponentTypeId_Test,ValidParameter){
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::PingTx", &tid));
}

TEST_F(GxfComponentTypeId_Test,InvalidClassName){
    GXF_ASSERT_EQ((GxfComponentTypeId(context, "invalid_component", &tid)),GXF_FACTORY_UNKNOWN_CLASS_NAME);
}

TEST_F(GxfComponentTypeId_Test,InvalidContext){
    GXF_ASSERT_EQ((GxfComponentTypeId(kNullContext, "nvidia::gxf::PingTx", &tid)),GXF_CONTEXT_INVALID);
}
