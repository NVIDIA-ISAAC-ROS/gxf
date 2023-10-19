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

class GxfParameterGet1DFloat64Vector_Test : public ::testing::Test {
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

TEST_F(GxfParameterGet1DFloat64Vector_Test, ValidParameter) {
  double float_1d_vector[4] = {0, -1200.6, 101, 32000.524};
  GXF_ASSERT_SUCCESS(
      GxfParameterSet1DFloat64Vector(context, cid, "float_1d_vector", float_1d_vector, 4));

  GXF_ASSERT_SUCCESS(
      GxfParameterGet1DFloat64VectorInfo(context, cid, "float_1d_vector", length_ptr));
  GXF_ASSERT_EQ(length, 4);
  double value[length];
  GXF_ASSERT_SUCCESS(
      GxfParameterGet1DFloat64Vector(context, cid, "float_1d_vector", value, length_ptr));
  // verify the array values fetched from api
  for (uint32_t i = 0; i < length; i++) {
    printf("value = %f  vector = %f", value[i], float_1d_vector[i]);
    GXF_ASSERT_EQ(value[i], float_1d_vector[i]);
  }
}

TEST_F(GxfParameterGet1DFloat64Vector_Test, InvalidContext) {
  double float_1d_vector[4] = {1, 5.009, 6529, -10};
  GXF_ASSERT_SUCCESS(
      GxfParameterSet1DFloat64Vector(context, cid, "float_1d_vector", float_1d_vector, 4));

  double value[0] = {};
  GXF_ASSERT_EQ(
      GxfParameterGet1DFloat64Vector(kNullContext, cid, "float_1d_vector", value, length_ptr),
      GXF_CONTEXT_INVALID);
}

TEST_F(GxfParameterGet1DFloat64Vector_Test, InvalidComponentID) {
  double float_1d_vector[4] = {11.6584, -54, 6865.00005, 1.08};
  GXF_ASSERT_SUCCESS(
      GxfParameterSet1DFloat64Vector(context, cid, "float_1d_vector", float_1d_vector, 4));

  GXF_ASSERT_EQ(
      GxfParameterGet1DFloat64VectorInfo(context, kNullUid, "float_1d_vector", length_ptr),
      GXF_PARAMETER_NOT_FOUND);

  double value[0] = {};
  GXF_ASSERT_EQ(
      GxfParameterGet1DFloat64Vector(context, kNullUid, "float_1d_vector", value, length_ptr),
      GXF_PARAMETER_NOT_FOUND);
}

TEST_F(GxfParameterGet1DFloat64Vector_Test, InvalidParameterName) {
  double float_1d_vector[4] = {10001, 0.05, 65, -108};
  GXF_ASSERT_SUCCESS(
      GxfParameterSet1DFloat64Vector(context, cid, "float_1d_vector", float_1d_vector, 4));

  GXF_ASSERT_SUCCESS(
      GxfParameterGet1DFloat64VectorInfo(context, cid, "float_1d_vector", length_ptr));
  double value[length];

  GXF_ASSERT_EQ(GxfParameterGet1DFloat64Vector(context, cid, "float_vector", value, length_ptr),
                GXF_PARAMETER_NOT_FOUND);
}

TEST_F(GxfParameterGet1DFloat64Vector_Test, InvalidParameterType) {
  uint64_t uint_1d_vector[4] = {129, 57, 6, 10};
  GXF_ASSERT_SUCCESS(
      GxfParameterSet1DUInt64Vector(context, cid, "uint_1d_vector", uint_1d_vector, 4));

  GXF_ASSERT_SUCCESS(GxfParameterGet1DUInt64VectorInfo(context, cid, "uint_1d_vector", length_ptr));

  double value[length];

  GXF_ASSERT_EQ(GxfParameterGet1DFloat64Vector(context, cid, "uint_1d_vector", value, length_ptr),
                GXF_PARAMETER_INVALID_TYPE);
}

TEST_F(GxfParameterGet1DFloat64Vector_Test, ArgumentNull) {
  double float_1d_vector[4] = {-1.1, 564, 0.007, 108};
  GXF_ASSERT_SUCCESS(
      GxfParameterSet1DFloat64Vector(context, cid, "float_1d_vector", float_1d_vector, 4));

  GXF_ASSERT_EQ(GxfParameterGet1DFloat64VectorInfo(context, cid, "float_1d_vector", length_ptr),
                GXF_SUCCESS);

  GXF_ASSERT_EQ(
      GxfParameterGet1DFloat64Vector(context, cid, "float_1d_vector", nullptr, length_ptr),
      GXF_ARGUMENT_NULL);
}

TEST_F(GxfParameterGet1DFloat64Vector_Test, EmptyParameterValue) {
  double float_1d_vector[0] = {};
  GXF_ASSERT_SUCCESS(
      GxfParameterSet1DFloat64Vector(context, cid, "float_1d_vector", float_1d_vector, 0));

  GXF_ASSERT_SUCCESS(GxfParameterGet1DFloat64VectorInfo(context, cid, "float_1d_vector", length_ptr));
  GXF_ASSERT_EQ(length, 0);

  double value[0] = {};
  GXF_ASSERT_SUCCESS(GxfParameterGet1DFloat64Vector(context, cid, "float_1d_vector", value, length_ptr));

}