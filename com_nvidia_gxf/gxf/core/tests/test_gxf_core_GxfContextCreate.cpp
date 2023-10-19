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

TEST(GxfCreateContext_Test, context) {
  gxf_context_t context = kNullContext;
  GXF_ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
}

TEST(GxfCreateContext_Test, MultipleContext) {
  int max = 1000;
  gxf_context_t context[max];
  for (int i = 0; i < max; i++) {
    GXF_ASSERT_EQ(GxfContextCreate(&context[i]), GXF_SUCCESS);
    }
  for (int i = 0; i < max; i++) {
    GxfContextDestroy(context[i]);
    }
}

TEST(GxfCreateContext_Test, NullArgumentZero) {
  GXF_ASSERT_EQ(GxfContextCreate(0), GXF_ARGUMENT_NULL);
}

TEST(GxfCreateContext_Test, KNullContext) {
  GXF_ASSERT_EQ(GxfContextCreate(kNullContext), GXF_ARGUMENT_NULL);
}