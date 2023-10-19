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
#include "yaml-cpp/yaml.h"

namespace {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

}  // namespace

class GxfParameterGetAsYamlNode_Test : public ::testing::Test {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
    GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::BlockMemoryPool", &tid));
    GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "test", &cid));
  }

  void TearDown() {
    GXF_ASSERT_SUCCESS(GxfEntityDestroy(context, eid));
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
  }

 protected:
  gxf_context_t context = kNullContext;
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  gxf_uid_t eid = kNullUid;
  const GxfEntityCreateInfo entity_create_info = {0};
  gxf_tid_t tid = GxfTidNull();
  gxf_uid_t cid = kNullUid;
  YAML::Node value;
};

TEST_F(GxfParameterGetAsYamlNode_Test, ValidParameterInt64) {
  int64_t int64_var = 101;
  GXF_ASSERT_SUCCESS(GxfParameterSetInt64(context, cid, "int64_var", int64_var));

  GXF_ASSERT_SUCCESS(GxfParameterGetAsYamlNode(context, cid, "int64_var", &value));
  GXF_ASSERT_EQ(value.as<int64_t>(), int64_var);
}

TEST_F(GxfParameterGetAsYamlNode_Test, InvalidContext) {
  GXF_ASSERT_EQ(GxfParameterGetAsYamlNode(kNullContext, cid, "int64_var", &value),
                GXF_CONTEXT_INVALID);
}

TEST_F(GxfParameterGetAsYamlNode_Test, InvalidUid) {
  GXF_ASSERT_EQ(GxfParameterGetAsYamlNode(context, kNullUid, "int64_var", &value),
                GXF_PARAMETER_NOT_FOUND);
}

TEST_F(GxfParameterGetAsYamlNode_Test, InvalidKey) {
  int64_t int64_var = 101;
  GXF_ASSERT_SUCCESS(GxfParameterSetInt64(context, cid, "int64_var", int64_var));
  GXF_ASSERT_EQ(GxfParameterGetAsYamlNode(context, cid, "int64_invalid_var", &value),
                GXF_PARAMETER_NOT_FOUND);
}
