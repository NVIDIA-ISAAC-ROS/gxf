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

class GxfParameterSet1DInt64Vector_Test : public ::testing::Test {
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
  uint64_t length;
  uint64_t* length_ptr = &length;
};

TEST_F(GxfParameterSet1DInt64Vector_Test, ValidParameter) {
  int64_t int64_1d_vector[4] = {0, -1200, 101, 32000};
  uint64_t vector_length = sizeof(int64_1d_vector) / sizeof(int64_1d_vector[0]);
  GXF_ASSERT_SUCCESS(GxfParameterSet1DInt64Vector(context, cid, "int64_1d_vector",
                                                    int64_1d_vector, vector_length));

  GXF_ASSERT_SUCCESS(
      GxfParameterGet1DInt64VectorInfo(context, cid, "int64_1d_vector", length_ptr));
  GXF_ASSERT_EQ(length, vector_length);
  int64_t value[length];
  GXF_ASSERT_SUCCESS(
      GxfParameterGet1DInt64Vector(context, cid, "int64_1d_vector", value, length_ptr));
  for (uint32_t i = 0; i < length; i++) { GXF_ASSERT_EQ(value[i], int64_1d_vector[i]); }
}

TEST_F(GxfParameterSet1DInt64Vector_Test, InvalidContext) {
  int64_t int64_1d_vector[4] = {1, 5, 6529, -10};
  uint64_t vector_length = sizeof(int64_1d_vector) / sizeof(int64_1d_vector[0]);
  GXF_ASSERT_EQ(GxfParameterSet1DInt64Vector(kNullContext, cid, "int64_1d_vector",
                                               int64_1d_vector, vector_length),
                GXF_CONTEXT_INVALID);
}

TEST_F(GxfParameterSet1DInt64Vector_Test, NullValue) {
  GXF_ASSERT_EQ(GxfParameterSet1DInt64Vector(context, cid, "int64_1d_vector", nullptr, 4),
                GXF_ARGUMENT_NULL);
}

TEST_F(GxfParameterSet1DInt64Vector_Test, OverwritingValidParameterType) {
  int64_t int64_1d_vector_1[4] = {10001, 0, 65, -108};
  GXF_ASSERT_SUCCESS(
      GxfParameterSet1DInt64Vector(context, cid, "int64_1d_vector", int64_1d_vector_1,
                                     sizeof(int64_1d_vector_1) / sizeof(int64_1d_vector_1[0])));
  GXF_ASSERT_SUCCESS(
      GxfParameterGet1DInt64VectorInfo(context, cid, "int64_1d_vector", length_ptr));
  int64_t value_1[length];
  GXF_ASSERT_SUCCESS(
      GxfParameterGet1DInt64Vector(context, cid, "int64_1d_vector", value_1, length_ptr));
  for (uint32_t i = 0; i < length; i++) { GXF_ASSERT_EQ(value_1[i], int64_1d_vector_1[i]); }

  int64_t int64_1d_vector_2[6] = {10355, 0, 65, -108, -137, 124};
  GXF_ASSERT_SUCCESS(
      GxfParameterSet1DInt64Vector(context, cid, "int64_1d_vector", int64_1d_vector_2,
                                     sizeof(int64_1d_vector_2) / sizeof(int64_1d_vector_2[0])));
  GXF_ASSERT_SUCCESS(
      GxfParameterGet1DInt64VectorInfo(context, cid, "int64_1d_vector", length_ptr));
  int64_t value_2[length];
  GXF_ASSERT_SUCCESS(
      GxfParameterGet1DInt64Vector(context, cid, "int64_1d_vector", value_2, length_ptr));
  for (uint32_t i = 0; i < length; i++) { GXF_ASSERT_EQ(value_2[i], int64_1d_vector_2[i]); }
}

TEST_F(GxfParameterSet1DInt64Vector_Test, SettingInvalidParameterType) {
  double float64_1d_vector[4] = {10001.34, 0.56, -65, -0.108};
  GXF_ASSERT_SUCCESS(
      GxfParameterSet1DFloat64Vector(context, cid, "float64_1d_vector", float64_1d_vector,
                                     sizeof(float64_1d_vector) / sizeof(float64_1d_vector[0])));

  int64_t int64_1d_vector[4] = {10355, 0, 65, -108};
  GXF_ASSERT_EQ(
      GxfParameterSet1DInt64Vector(context, cid, "float64_1d_vector", int64_1d_vector,
                                     sizeof(int64_1d_vector) / sizeof(int64_1d_vector[0])), GXF_PARAMETER_INVALID_TYPE);
}

TEST_F(GxfParameterSet1DInt64Vector_Test, BoundaryValue) {
  int64_t int64_1d_vector[2] = {INT64_MAX, INT64_MIN};
  uint64_t vector_length = sizeof(int64_1d_vector) / sizeof(int64_1d_vector[0]);
  GXF_ASSERT_SUCCESS(GxfParameterSet1DInt64Vector(context, cid, "int64_1d_vector",
                                                    int64_1d_vector, vector_length));

  GXF_ASSERT_SUCCESS(
      GxfParameterGet1DInt64VectorInfo(context, cid, "int64_1d_vector", length_ptr));
  GXF_ASSERT_EQ(length, vector_length);
  int64_t value[length];
  GXF_ASSERT_SUCCESS(
      GxfParameterGet1DInt64Vector(context, cid, "int64_1d_vector", value, length_ptr));
  for (uint32_t i = 0; i < length; i++) { GXF_ASSERT_EQ(value[i], int64_1d_vector[i]); }
}
