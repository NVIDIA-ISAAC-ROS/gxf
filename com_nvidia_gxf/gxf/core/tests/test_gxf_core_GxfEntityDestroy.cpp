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
#include <thread>
#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"

TEST(GxfEntityDestroy,valid_context)
{
   gxf_context_t context = kNullContext;
   gxf_uid_t eid = kNullUid;
   ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
   const GxfEntityCreateInfo entity_create_info = {0};
   ASSERT_EQ(GxfCreateEntity(context,&entity_create_info, &eid),GXF_SUCCESS);
   ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
   ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(GxfEntityDestroy,invalid_context)
{
   gxf_context_t context = kNullContext;
   gxf_uid_t eid = kNullUid;
   ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
   const GxfEntityCreateInfo entity_create_info = {0};
   ASSERT_EQ(GxfCreateEntity(context,&entity_create_info, &eid),GXF_SUCCESS);
   GXF_ASSERT_EQ(GxfEntityDestroy(NULL, eid),GXF_CONTEXT_INVALID);
   ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(GxfEntityDestroy,invalid_eid)
{
   gxf_context_t context = kNullContext;
   gxf_uid_t eid = kNullUid;
   ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
   const GxfEntityCreateInfo entity_create_info = {0};
   ASSERT_EQ(GxfCreateEntity(context,&entity_create_info, &eid),GXF_SUCCESS);
   GXF_ASSERT_EQ(GxfEntityDestroy(context,kNullUid),GXF_QUERY_NOT_FOUND);
   ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}
