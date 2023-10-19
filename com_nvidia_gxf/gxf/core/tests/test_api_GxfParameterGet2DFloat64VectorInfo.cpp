/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"

namespace {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

}  // namespace

class GxfParameterGet2DFloat64VectorInfo_Test : public ::testing::Test {
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
  uint64_t height;
  uint64_t* height_ptr = &height;
  uint64_t width;
  uint64_t* width_ptr = &width;
};

TEST_F(GxfParameterGet2DFloat64VectorInfo_Test, ValidParameter) {
  double float_1d_vector_1[] = {0.01, 5.2, -4, 55.5, -94.99};
  double float_1d_vector_2[] = {26, -71, -0.5, 666.6, 3600};
  double* float_2d_vector[] = {float_1d_vector_1, float_1d_vector_2};

  GXF_ASSERT_SUCCESS(
      GxfParameterSet2DFloat64Vector(context, cid, "float_2d_vector", float_2d_vector, 2, 5));

  GXF_ASSERT_SUCCESS(
      GxfParameterGet2DFloat64VectorInfo(context, cid, "float_2d_vector", height_ptr, width_ptr));
  GXF_ASSERT_EQ(height, 2);
  GXF_ASSERT_EQ(width, 5);
}

TEST_F(GxfParameterGet2DFloat64VectorInfo_Test, InvalidContext) {
  double float_1d_vector_1[] = {0.01, 5.2, -4, 55.5, -94.99};
  double float_1d_vector_2[] = {26, -71, -0.5, 666.6, 3600};
  double* float_2d_vector[] = {float_1d_vector_1, float_1d_vector_2};

  GXF_ASSERT_SUCCESS(
      GxfParameterSet2DFloat64Vector(context, cid, "float_2d_vector", float_2d_vector, 2, 5));

  GXF_ASSERT_EQ(GxfParameterGet2DFloat64VectorInfo(kNullContext, cid, "float_2d_vector", height_ptr,
                                                   width_ptr),
                GXF_CONTEXT_INVALID);
}

TEST_F(GxfParameterGet2DFloat64VectorInfo_Test, InvalidComponentID) {
  double float_1d_vector_1[] = {0.01, 5.2, -4, 55.5, -94.99};
  double float_1d_vector_2[] = {26, -71, -0.5, 666.6, 3600};
  double* float_2d_vector[] = {float_1d_vector_1, float_1d_vector_2};

  GXF_ASSERT_SUCCESS(
      GxfParameterSet2DFloat64Vector(context, cid, "float_2d_vector", float_2d_vector, 2, 5));

  GXF_ASSERT_EQ(GxfParameterGet2DFloat64VectorInfo(context, kNullUid, "float_2d_vector", height_ptr,
                                                   width_ptr),
                GXF_PARAMETER_NOT_FOUND);
}

TEST_F(GxfParameterGet2DFloat64VectorInfo_Test, InvalidParameterName) {
  double float_1d_vector_1[] = {0.01, 5.2, -4, 55.5, -94.99};
  double float_1d_vector_2[] = {26, -71, -0.5, 666.6, 3600};
  double* float_2d_vector[] = {float_1d_vector_1, float_1d_vector_2};

  GXF_ASSERT_SUCCESS(
      GxfParameterSet2DFloat64Vector(context, cid, "float_2d_vector", float_2d_vector, 2, 5));

  GXF_ASSERT_EQ(GxfParameterGet2DFloat64VectorInfo(context, cid, "vector", height_ptr, width_ptr),
                GXF_PARAMETER_NOT_FOUND);
}

TEST_F(GxfParameterGet2DFloat64VectorInfo_Test, InvalidParameterType) {
  int64_t int_1d_vector_1[] = {1, 52, -4, 55, -9499};
  int64_t int_1d_vector_2[] = {26, -71, -5, 666, 3600};
  int64_t* int_2d_vector[] = {int_1d_vector_1, int_1d_vector_2};

  GXF_ASSERT_SUCCESS(
      GxfParameterSet2DInt64Vector(context, cid, "int_2d_vector", int_2d_vector, 2, 5));

  GXF_ASSERT_EQ(GxfParameterGet2DFloat64VectorInfo(context, cid, "int_2d_vector", height_ptr, width_ptr),
                GXF_PARAMETER_INVALID_TYPE);
}

TEST_F(GxfParameterGet2DFloat64VectorInfo_Test, ArgumentNull) {
  double float_1d_vector_1[] = {0.01, 5.2, -4, 55.5, -94.99};
  double float_1d_vector_2[] = {26, -71, -0.5, 666.6, 3600};
  double* float_2d_vector[] = {float_1d_vector_1, float_1d_vector_2};

  GXF_ASSERT_SUCCESS(
      GxfParameterSet2DFloat64Vector(context, cid, "float_2d_vector", float_2d_vector, 2, 5));

  GXF_ASSERT_EQ(GxfParameterGet2DFloat64VectorInfo(context, cid, "float_2d_vector", nullptr, width_ptr),
                GXF_ARGUMENT_NULL);
}

TEST_F(GxfParameterGet2DFloat64VectorInfo_Test, EmptyParameterValue) {
  double float_1d_vector_1[] = {};
  double float_1d_vector_2[] = {};
  double* float_2d_vector[] = {float_1d_vector_1, float_1d_vector_2};

  GXF_ASSERT_SUCCESS(
      GxfParameterSet2DFloat64Vector(context, cid, "float_2d_vector", float_2d_vector, 2, 0));
  GXF_ASSERT_SUCCESS(
      GxfParameterGet2DFloat64VectorInfo(context, cid, "float_2d_vector", height_ptr, width_ptr));
}