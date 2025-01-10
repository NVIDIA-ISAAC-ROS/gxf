/*
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

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
const char* kStdExtensionFilename = "gxf/std/libgxf_std.so";
const char* kTestExtensionFilename = "gxf/test/extensions/libgxf_test.so";
const char* kSampleExtensionFilename = "gxf/sample/libgxf_sample.so";
}  // namespace

class GxfGraphLoadFileTest : public ::testing::Test {
 protected:
  void SetUp() override {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info_std));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info_test));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info_sample));
  }

  gxf_result_t SetUpUnregisteredParamTests(gxf_uid_t* component) {
    const char* kGraphFileName =
        "gxf/core/tests/apps/test_app_GxfGraphLoadFile_unregistered_params.yaml";
    GXF_ASSERT_EQ(GxfGraphLoadFile(context, kGraphFileName), GXF_SUCCESS);

    gxf_uid_t* entities = new gxf_uid_t[32];
    uint64_t entities_count = 32;
    GXF_ASSERT_SUCCESS(GxfEntityFindAll(context, &entities_count, entities));
    GXF_ASSERT_EQ(entities_count, 1);

    GXF_ASSERT_SUCCESS(
        GxfComponentFind(context, entities[0], GxfTidNull(), "step", nullptr, component));
    delete[] entities;
    return GXF_SUCCESS;
  }

  gxf_context_t context = kNullContext;
  const GxfLoadExtensionsInfo info_std{&kStdExtensionFilename, 1, nullptr, 0, nullptr};
  const GxfLoadExtensionsInfo info_test{&kTestExtensionFilename, 1, nullptr, 0, nullptr};
  const GxfLoadExtensionsInfo info_sample{&kSampleExtensionFilename, 1, nullptr, 0, nullptr};
};

TEST_F(GxfGraphLoadFileTest, Valid) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphLoadFile_valid.yaml";
  GXF_ASSERT_EQ(GxfGraphLoadFile(context, kGraphFileName), GXF_SUCCESS);
}

TEST_F(GxfGraphLoadFileTest, LoadMultipleFiles) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphLoadFile_valid.yaml";
  GXF_ASSERT_EQ(GxfGraphLoadFile(context, kGraphFileName), GXF_SUCCESS);
  GXF_ASSERT_EQ(
      GxfGraphLoadFile(context,
                       "gxf/core/tests/apps/test_app_GxfGraphLoadFile_unregistered_params.yaml"),
      GXF_SUCCESS);
}

TEST_F(GxfGraphLoadFileTest, InvalidContext) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphLoadFile_valid.yaml";
  GXF_ASSERT_EQ(GxfGraphLoadFile(kNullContext, kGraphFileName), GXF_CONTEXT_INVALID);
  GXF_ASSERT_EQ(GxfGraphLoadFile(nullptr, kGraphFileName), GXF_CONTEXT_INVALID);
}

TEST_F(GxfGraphLoadFileTest, ComponentNotSequence) {
  const char* kGraphFileName =
      "gxf/core/tests/apps/test_app_GxfGraphLoadFile_component_not_seq.yaml";
  GXF_ASSERT_EQ(GxfGraphLoadFile(context, kGraphFileName), GXF_FAILURE);
}

TEST_F(GxfGraphLoadFileTest, InvalidFilePath) {
  const char* kGraphFileName = "/wrong/path.yaml";
  GXF_ASSERT_EQ(GxfGraphLoadFile(context, kGraphFileName), GXF_FAILURE);
}

TEST_F(GxfGraphLoadFileTest, ValidUnregisteredParamBool) {
  gxf_uid_t component;
  SetUpUnregisteredParamTests(&component);

  bool val;
  GXF_ASSERT_SUCCESS(GxfParameterGetBool(context, component, "test_bool", &val));
  GXF_ASSERT_EQ(val, true);
}
TEST_F(GxfGraphLoadFileTest, ValidUnregisteredParamInt64) {
  gxf_uid_t component;
  SetUpUnregisteredParamTests(&component);

  int64_t val;
  GXF_ASSERT_SUCCESS(GxfParameterGetInt64(context, component, "test_int", &val));
  GXF_ASSERT_EQ(val, 22);
}

TEST_F(GxfGraphLoadFileTest, ValidUnregisteredParamInt64WithMsTag) {
  gxf_uid_t component;
  SetUpUnregisteredParamTests(&component);

  int64_t val;
  GXF_ASSERT_SUCCESS(GxfParameterGetInt64(context, component, "test_int_with_ms_tag", &val));
  GXF_ASSERT_EQ(val, 2'000'000);
}

TEST_F(GxfGraphLoadFileTest, ValidUnregisteredParamFloat) {
  gxf_uid_t component;
  SetUpUnregisteredParamTests(&component);

  double val;
  GXF_ASSERT_SUCCESS(GxfParameterGetFloat64(context, component, "test_float", &val));
  GXF_ASSERT_EQ(val, 24.5);
}

TEST_F(GxfGraphLoadFileTest, ValidUnregisteredParamString) {
  gxf_uid_t component;
  SetUpUnregisteredParamTests(&component);

  const char* val;
  GXF_ASSERT_SUCCESS(GxfParameterGetStr(context, component, "test_string", &val));
  GXF_ASSERT_EQ(strcmp(val, "string"), 0);
}

TEST_F(GxfGraphLoadFileTest, InvalidYAMLFormat) {
  const char* kGraphFileName = "gxf/core/tests/apps/test_app_GxfGraphLoadFile_invalid_format.yaml";
  GXF_ASSERT_EQ(GxfGraphLoadFile(context, kGraphFileName), GXF_FAILURE);
}

TEST_F(GxfGraphLoadFileTest, InvalidComponentType) {
  const char* kGraphFileName =
      "gxf/core/tests/apps/test_app_GxfGraphLoadFile_invalid_component_type.yaml";
  GXF_ASSERT_EQ(GxfGraphLoadFile(context, kGraphFileName), GXF_FACTORY_UNKNOWN_CLASS_NAME);
}

TEST_F(GxfGraphLoadFileTest, InvalidParameterValue) {
  const char* kGraphFileName =
      "gxf/core/tests/apps/test_app_GxfGraphLoadFile_invalid_parameter_value.yaml";
  GXF_ASSERT_EQ(GxfGraphLoadFile(context, kGraphFileName), GXF_PARAMETER_PARSER_ERROR);
}

TEST_F(GxfGraphLoadFileTest, InvalidEntityReference) {
  const char* kGraphFileName =
      "gxf/core/tests/apps/test_app_GxfGraphLoadFile_invalid_entity_ref.yaml";
  GXF_ASSERT_EQ(GxfGraphLoadFile(context, kGraphFileName), GXF_ENTITY_NOT_FOUND);
}

TEST_F(GxfGraphLoadFileTest, InvalidComponentReference) {
  const char* kGraphFileName =
      "gxf/core/tests/apps/test_app_GxfGraphLoadFile_invalid_component_ref.yaml";
  GXF_ASSERT_EQ(GxfGraphLoadFile(context, kGraphFileName), GXF_ENTITY_COMPONENT_NOT_FOUND);
}

TEST_F(GxfGraphLoadFileTest, InvalidParameterChildType) {
  const char* kGraphFileName =
      "gxf/core/tests/apps/test_app_GxfGraphLoadFile_parameter_not_a_map.yaml";
  GXF_ASSERT_EQ(GxfGraphLoadFile(context, kGraphFileName), GXF_INVALID_DATA_FORMAT);
}
