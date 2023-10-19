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


class GxfComponentTypeName_Test : public ::testing::Test {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::PingTx", &tid));
  }

  void TearDown() {
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
  }

 protected:
  gxf_context_t context = kNullContext;
  gxf_tid_t tid = GxfTidNull();
  gxf_tid_t null_tid = GxfTidNull();
  const char* name;
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
};

TEST_F(GxfComponentTypeName_Test,ValidParameter){
    GXF_ASSERT_SUCCESS(GxfComponentTypeName(context, tid, &name));
}

TEST_F(GxfComponentTypeName_Test,InvalidTid){
    GXF_ASSERT_EQ((GxfComponentTypeName(context, null_tid, &name)),GXF_FAILURE);
}

TEST_F(GxfComponentTypeName_Test,Nullpointer){
    GXF_ASSERT_EQ((GxfComponentTypeName(context, tid, nullptr)),GXF_NULL_POINTER);
}
