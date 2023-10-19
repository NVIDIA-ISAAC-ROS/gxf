/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <chrono>
#include <climits>
#include <cstring>
#include <iostream>
#include <thread>

#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"

namespace {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

}  // namespace

class GxfParameterGet1DInt64Vector_Test : public ::testing::Test {
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

TEST_F(GxfParameterGet1DInt64Vector_Test, ValidParameter) {
  int64_t int_1d_vector[4] = {0, -1200, 101, 320004};
  GXF_ASSERT_SUCCESS(GxfParameterSet1DInt64Vector(context, cid, "int_1d_vector", int_1d_vector, 4));

  GXF_ASSERT_SUCCESS(GxfParameterGet1DInt64VectorInfo(context, cid, "int_1d_vector", length_ptr));
  GXF_ASSERT_EQ(length, 4);
  int64_t value[length];
  GXF_ASSERT_SUCCESS(
      GxfParameterGet1DInt64Vector(context, cid, "int_1d_vector", value, length_ptr));
  // verify the array values fetched from api
  for (uint32_t i = 0; i < length; i++) {
    GXF_ASSERT_EQ(value[i], int_1d_vector[i]);
  }
}

TEST_F(GxfParameterGet1DInt64Vector_Test, InvalidContext) {
  int64_t int_1d_vector[4] = {1, -509, 6529, -10};
  GXF_ASSERT_SUCCESS(
      GxfParameterSet1DInt64Vector(context, cid, "int_1d_vector", int_1d_vector, 4));

  int64_t value[0] = {};
  GXF_ASSERT_EQ(
      GxfParameterGet1DInt64Vector(kNullContext, cid, "int_1d_vector", value, length_ptr),
      GXF_CONTEXT_INVALID);
}

TEST_F(GxfParameterGet1DInt64Vector_Test, InvalidComponentID) {
  int64_t int_1d_vector[4] = {0, -54, 6865, 18};
  GXF_ASSERT_SUCCESS(
      GxfParameterSet1DInt64Vector(context, cid, "int_1d_vector", int_1d_vector, 4));

  GXF_ASSERT_EQ(GxfParameterGet1DInt64VectorInfo(context, kNullUid, "int_1d_vector", length_ptr),
                GXF_PARAMETER_NOT_FOUND);

  int64_t value[0] = {};
  GXF_ASSERT_EQ(
      GxfParameterGet1DInt64Vector(context, kNullUid, "int_1d_vector", value, length_ptr),
      GXF_PARAMETER_NOT_FOUND);
}

TEST_F(GxfParameterGet1DInt64Vector_Test, InvalidParameterName) {
  int64_t int_1d_vector[4] = {10001, 5, 65, -108};
  GXF_ASSERT_SUCCESS(
      GxfParameterSet1DInt64Vector(context, cid, "int_1d_vector", int_1d_vector, 4));

  GXF_ASSERT_SUCCESS(GxfParameterGet1DInt64VectorInfo(context, cid, "int_1d_vector", length_ptr));
  int64_t value[length];

  GXF_ASSERT_EQ(GxfParameterGet1DInt64Vector(context, cid, "int_vector", value, length_ptr),
                GXF_PARAMETER_NOT_FOUND);
}

TEST_F(GxfParameterGet1DInt64Vector_Test, InvalidParameterType) {
  double float_1d_vector[4] = {12.9, -5.7, 0.0006, 18640};
  GXF_ASSERT_SUCCESS(
      GxfParameterSet1DFloat64Vector(context, cid, "float_1d_vector", float_1d_vector, 4));

  GXF_ASSERT_SUCCESS(GxfParameterGet1DFloat64VectorInfo(context, cid, "float_1d_vector", length_ptr));

  int64_t value[length];

  GXF_ASSERT_EQ(GxfParameterGet1DInt64Vector(context, cid, "float_1d_vector", value, length_ptr),
                GXF_PARAMETER_INVALID_TYPE);
}

TEST_F(GxfParameterGet1DInt64Vector_Test, ArgumentNull) {
  int64_t int_1d_vector[4] = {-11, 564, -777, 8};
  GXF_ASSERT_SUCCESS(
      GxfParameterSet1DInt64Vector(context, cid, "int_1d_vector", int_1d_vector, 4));

  GXF_ASSERT_EQ(GxfParameterGet1DInt64VectorInfo(context, cid, "int_1d_vector", length_ptr),
                GXF_SUCCESS);

  GXF_ASSERT_EQ(GxfParameterGet1DInt64Vector(context, cid, "int_1d_vector", nullptr, length_ptr),
                GXF_ARGUMENT_NULL);
}

TEST_F(GxfParameterGet1DInt64Vector_Test, EmptyParameterValue) {
  int64_t int_1d_vector[0] = {};
  GXF_ASSERT_SUCCESS(
      GxfParameterSet1DInt64Vector(context, cid, "int_1d_vector", int_1d_vector, 0));

  GXF_ASSERT_SUCCESS(GxfParameterGet1DInt64VectorInfo(context, cid, "int_1d_vector", length_ptr));
  GXF_ASSERT_EQ(length, 0);

  int64_t value[0] = {};
  GXF_ASSERT_SUCCESS(
      GxfParameterGet1DInt64Vector(context, cid, "int_1d_vector", value, length_ptr));
}