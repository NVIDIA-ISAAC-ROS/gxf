/*
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"

TEST(GxfParameterTypeStr, Custom) {
  gxf_parameter_type_t type = gxf_parameter_type_t::GXF_PARAMETER_TYPE_CUSTOM;
  GXF_ASSERT_EQ(strcmp(GxfParameterTypeStr(type), "GXF_PARAMETER_TYPE_CUSTOM"), 0);
}

TEST(GxfParameterTypeStr, Handle) {
  gxf_parameter_type_t type = gxf_parameter_type_t::GXF_PARAMETER_TYPE_HANDLE;
  GXF_ASSERT_EQ(strcmp(GxfParameterTypeStr(type), "GXF_PARAMETER_TYPE_HANDLE"), 0);
}

TEST(GxfParameterTypeStr, String) {
  gxf_parameter_type_t type = gxf_parameter_type_t::GXF_PARAMETER_TYPE_STRING;
  GXF_ASSERT_EQ(strcmp(GxfParameterTypeStr(type), "GXF_PARAMETER_TYPE_STRING"), 0);
}

TEST(GxfParameterTypeStr, Int64) {
  gxf_parameter_type_t type = gxf_parameter_type_t::GXF_PARAMETER_TYPE_INT64;
  GXF_ASSERT_EQ(strcmp(GxfParameterTypeStr(type), "GXF_PARAMETER_TYPE_INT64"), 0);
}

TEST(GxfParameterTypeStr, UInt64) {
  gxf_parameter_type_t type = gxf_parameter_type_t::GXF_PARAMETER_TYPE_UINT64;
  GXF_ASSERT_EQ(strcmp(GxfParameterTypeStr(type), "GXF_PARAMETER_TYPE_UINT64"), 0);
}
TEST(GxfParameterTypeStr, Float64) {
  gxf_parameter_type_t type = gxf_parameter_type_t::GXF_PARAMETER_TYPE_FLOAT64;
  GXF_ASSERT_EQ(strcmp(GxfParameterTypeStr(type), "GXF_PARAMETER_TYPE_FLOAT64"), 0);
}

TEST(GxfParameterTypeStr, Bool) {
  gxf_parameter_type_t type = gxf_parameter_type_t::GXF_PARAMETER_TYPE_BOOL;
  GXF_ASSERT_EQ(strcmp(GxfParameterTypeStr(type), "GXF_PARAMETER_TYPE_BOOL"), 0);
}

TEST(GxfParameterTypeStr, Int32) {
  gxf_parameter_type_t type = gxf_parameter_type_t::GXF_PARAMETER_TYPE_INT32;
  GXF_ASSERT_EQ(strcmp(GxfParameterTypeStr(type), "GXF_PARAMETER_TYPE_INT32"), 0);
}

TEST(GxfParameterTypeStr, File) {
  gxf_parameter_type_t type = gxf_parameter_type_t::GXF_PARAMETER_TYPE_FILE;
  GXF_ASSERT_EQ(strcmp(GxfParameterTypeStr(type), "GXF_PARAMETER_TYPE_FILE"), 0);
}

TEST(GxfParameterTypeStr, Int8) {
  gxf_parameter_type_t type = gxf_parameter_type_t::GXF_PARAMETER_TYPE_INT8;
  GXF_ASSERT_EQ(strcmp(GxfParameterTypeStr(type), "GXF_PARAMETER_TYPE_INT8"), 0);
}

TEST(GxfParameterTypeStr, Int16) {
  gxf_parameter_type_t type = gxf_parameter_type_t::GXF_PARAMETER_TYPE_INT16;
  GXF_ASSERT_EQ(strcmp(GxfParameterTypeStr(type), "GXF_PARAMETER_TYPE_INT16"), 0);
}

TEST(GxfParameterTypeStr, Uint8) {
  gxf_parameter_type_t type = gxf_parameter_type_t::GXF_PARAMETER_TYPE_UINT8;
  GXF_ASSERT_EQ(strcmp(GxfParameterTypeStr(type), "GXF_PARAMETER_TYPE_UINT8"), 0);
}

TEST(GxfParameterTypeStr, Uint16) {
  gxf_parameter_type_t type = gxf_parameter_type_t::GXF_PARAMETER_TYPE_UINT16;
  GXF_ASSERT_EQ(strcmp(GxfParameterTypeStr(type), "GXF_PARAMETER_TYPE_UINT16"), 0);
}

TEST(GxfParameterTypeStr, Uint32) {
  gxf_parameter_type_t type = gxf_parameter_type_t::GXF_PARAMETER_TYPE_UINT32;
  GXF_ASSERT_EQ(strcmp(GxfParameterTypeStr(type), "GXF_PARAMETER_TYPE_UINT32"), 0);
}

TEST(GxfParameterTypeStr, Float32) {
  gxf_parameter_type_t type = gxf_parameter_type_t::GXF_PARAMETER_TYPE_FLOAT32;
  GXF_ASSERT_EQ(strcmp(GxfParameterTypeStr(type), "GXF_PARAMETER_TYPE_FLOAT32"), 0);
}

TEST(GxfParameterTypeStr, Complex64) {
  gxf_parameter_type_t type = gxf_parameter_type_t::GXF_PARAMETER_TYPE_COMPLEX64;
  GXF_ASSERT_EQ(strcmp(GxfParameterTypeStr(type), "GXF_PARAMETER_TYPE_COMPLEX64"), 0);
}

TEST(GxfParameterTypeStr, Complex128) {
  gxf_parameter_type_t type = gxf_parameter_type_t::GXF_PARAMETER_TYPE_COMPLEX128;
  GXF_ASSERT_EQ(strcmp(GxfParameterTypeStr(type), "GXF_PARAMETER_TYPE_COMPLEX128"), 0);
}
