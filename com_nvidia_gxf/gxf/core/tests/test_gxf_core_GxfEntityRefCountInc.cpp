/*
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

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

TEST(GxfEntityRefCountInc,valid_context)
{
    gxf_context_t context;
    gxf_uid_t eid = kNullUid;
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    const GxfEntityCreateInfo entity_create_info ={0};
    ASSERT_EQ(GxfCreateEntity(context,&entity_create_info, &eid),GXF_SUCCESS);
    ASSERT_NE(eid,kNullUid);
    GXF_ASSERT_SUCCESS(GxfEntityRefCountInc(context, eid));
    ASSERT_EQ(GxfContextDestroy(context),GXF_SUCCESS);
}

TEST(GxfEntityRefCountInc,null_context)
{
    gxf_context_t context;
    gxf_uid_t eid = kNullUid;
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    const GxfEntityCreateInfo entity_create_info ={0};
    GXF_ASSERT_SUCCESS(GxfCreateEntity(context,&entity_create_info, &eid));
    ASSERT_NE(eid,kNullUid);
    ASSERT_EQ(GxfEntityRefCountInc(NULL, eid),GXF_CONTEXT_INVALID);
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityRefCountInc,null_eid)
{
    gxf_context_t context;
    gxf_uid_t eid = kNullUid;
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    const GxfEntityCreateInfo entity_create_info ={0};
    GXF_ASSERT_SUCCESS(GxfCreateEntity(context,&entity_create_info, &eid));
    ASSERT_NE(eid,kNullUid);
    GXF_ASSERT_SUCCESS(GxfEntityRefCountInc(context,kNullUid));
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityRefCountInc,negative_eid)
{
    gxf_context_t context;
    gxf_uid_t eid = kNullUid;
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    const GxfEntityCreateInfo entity_create_info ={0};
     GXF_ASSERT_SUCCESS(GxfCreateEntity(context,&entity_create_info, &eid));
    ASSERT_NE(eid,kNullUid);
    GXF_ASSERT_SUCCESS(GxfEntityRefCountInc(context,-1));
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityRefCountInc,zero_eid_value)
{
    gxf_context_t context;
    gxf_uid_t eid = kNullUid;
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    const GxfEntityCreateInfo entity_create_info ={0};
     GXF_ASSERT_SUCCESS(GxfCreateEntity(context,&entity_create_info, &eid));
    ASSERT_NE(eid,kNullUid);
    GXF_ASSERT_SUCCESS(GxfEntityRefCountInc(context,0));
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}