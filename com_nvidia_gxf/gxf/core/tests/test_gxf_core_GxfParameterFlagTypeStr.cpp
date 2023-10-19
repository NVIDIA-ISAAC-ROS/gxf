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


TEST(GxfParameterFlagsTypeStr, FlagsNone) {
  gxf_parameter_flags_t_ type = gxf_parameter_flags_t_::GXF_PARAMETER_FLAGS_NONE;
  GXF_ASSERT_EQ(strcmp(GxfParameterFlagTypeStr(type), "GXF_PARAMETER_FLAGS_NONE"), 0);
}

TEST(GxfParameterFlagsTypeStr, FlagsOptional) {
  gxf_parameter_flags_t_ type = gxf_parameter_flags_t_::GXF_PARAMETER_FLAGS_OPTIONAL;
  GXF_ASSERT_EQ(strcmp(GxfParameterFlagTypeStr(type), "GXF_PARAMETER_FLAGS_OPTIONAL"), 0);
}

TEST(GxfParameterFlagsTypeStr, FlagsDynamic) {
  gxf_parameter_flags_t_ type = gxf_parameter_flags_t_::GXF_PARAMETER_FLAGS_DYNAMIC;
  GXF_ASSERT_EQ(strcmp(GxfParameterFlagTypeStr(type), "GXF_PARAMETER_FLAGS_DYNAMIC"), 0);
}