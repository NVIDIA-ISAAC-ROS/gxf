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

class GxfParameterGet1DFloat64VectorInfo_Test : public ::testing::Test {
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

TEST_F(GxfParameterGet1DFloat64VectorInfo_Test, ValidParameter) {
  double float_1d_vector[4] = {101.1, 1200.5, 101.8, 121.0};
  GXF_ASSERT_SUCCESS(
      GxfParameterSet1DFloat64Vector(context, cid, "float64_vector", float_1d_vector, 4));

  GXF_ASSERT_SUCCESS(
      GxfParameterGet1DFloat64VectorInfo(context, cid, "float64_vector", length_ptr));
  GXF_ASSERT_EQ(length, 4);
}

TEST_F(GxfParameterGet1DFloat64VectorInfo_Test, InvalidContext) {
  double float_1d_vector[4] = {1.1, 5.4, 65.86, 108.0};
  GXF_ASSERT_SUCCESS(
      GxfParameterSet1DFloat64Vector(context, cid, "float64_vector", float_1d_vector, 4));

  GXF_ASSERT_EQ(GxfParameterGet1DFloat64VectorInfo(kNullContext, cid, "float64_vector", length_ptr),
                GXF_CONTEXT_INVALID);
}

TEST_F(GxfParameterGet1DFloat64VectorInfo_Test, InvalidComponentID) {
  double float_1d_vector[4] = {1.1, 5.4, 65.86, 108.0};
  GXF_ASSERT_SUCCESS(
      GxfParameterSet1DFloat64Vector(context, cid, "float64_vector", float_1d_vector, 4));

  GXF_ASSERT_EQ(GxfParameterGet1DFloat64VectorInfo(context, kNullUid, "float64_vector", length_ptr),
                GXF_PARAMETER_NOT_FOUND);
}

TEST_F(GxfParameterGet1DFloat64VectorInfo_Test, InvalidParameterName) {
  double float_1d_vector[4] = {1.1, 5.4, 65.86, 108.0};
  GXF_ASSERT_SUCCESS(
      GxfParameterSet1DFloat64Vector(context, cid, "float64_vector", float_1d_vector, 4));

  GXF_ASSERT_EQ(GxfParameterGet1DFloat64VectorInfo(context, cid, "vector", length_ptr),
                GXF_PARAMETER_NOT_FOUND);
}

TEST_F(GxfParameterGet1DFloat64VectorInfo_Test, InvalidParameterType) {
  int64_t int_1d_vector[4] = {129, 5, 65, 108};
  GXF_ASSERT_SUCCESS(GxfParameterSet1DInt64Vector(context, cid, "int64_vector", int_1d_vector, 4));

  GXF_ASSERT_EQ(GxfParameterGet1DFloat64VectorInfo(context, cid, "int64_vector", length_ptr),
                GXF_PARAMETER_INVALID_TYPE);
}

TEST_F(GxfParameterGet1DFloat64VectorInfo_Test, ArgumentNull) {
  double float_1d_vector[4] = {1.1, 5.4, 65.86, 108.0};
  GXF_ASSERT_SUCCESS(
      GxfParameterSet1DFloat64Vector(context, cid, "float64_vector", float_1d_vector, 4));

  GXF_ASSERT_EQ(GxfParameterGet1DFloat64VectorInfo(context, cid, "float64_vector", nullptr),
                GXF_ARGUMENT_NULL);
}

TEST_F(GxfParameterGet1DFloat64VectorInfo_Test, EmptyParameterValue) {
  double float_1d_vector[0] = {};
  GXF_ASSERT_SUCCESS(
      GxfParameterSet1DFloat64Vector(context, cid, "float64_vector", float_1d_vector, 0));

  GXF_ASSERT_SUCCESS(
      GxfParameterGet1DFloat64VectorInfo(context, cid, "float64_vector", length_ptr));
  GXF_ASSERT_EQ(length, 0);
}
