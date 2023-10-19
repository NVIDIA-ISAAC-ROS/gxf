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

TEST(GxfEntityRefCountDec,valid_context)
{
    gxf_context_t context;
    ASSERT_EQ(GxfContextCreate(&context),GXF_SUCCESS);
    gxf_uid_t eid = kNullUid;
    const GxfEntityCreateInfo entity_create_info ={0};
    GXF_ASSERT_SUCCESS(GxfCreateEntity(context,&entity_create_info, &eid));
    ASSERT_NE(eid,kNullUid);
    GXF_ASSERT_SUCCESS(GxfEntityRefCountInc(context,eid));
    GXF_ASSERT_SUCCESS(GxfEntityRefCountDec(context,eid));
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityRefCountDec,null_context)
{
    gxf_context_t context;
    gxf_uid_t eid = kNullUid;
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    const GxfEntityCreateInfo entity_create_info ={0};
    ASSERT_EQ(GxfCreateEntity(context,&entity_create_info, &eid),GXF_SUCCESS);
    ASSERT_NE(eid,kNullUid);
    GXF_ASSERT_SUCCESS(GxfEntityRefCountInc(context, eid));
    GXF_ASSERT_EQ(GxfEntityRefCountDec(NULL,eid),GXF_CONTEXT_INVALID);
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityRefCountDec,decrementing_ref_count_without_increamenting_ref_count)
{
    gxf_context_t context;
    gxf_uid_t eid = kNullUid;
    const GxfEntityCreateInfo entity_create_info ={0};
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfCreateEntity(context,&entity_create_info, &eid));
    ASSERT_NE(eid,kNullUid);
    GXF_ASSERT_EQ(GxfEntityRefCountDec(context, eid),GXF_REF_COUNT_NEGATIVE);
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityRefCountDec,null_eid)
{
    gxf_context_t context;
    gxf_uid_t eid = kNullUid;
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    const GxfEntityCreateInfo entity_create_info ={0};
    GXF_ASSERT_SUCCESS(GxfCreateEntity(context,&entity_create_info, &eid));
    ASSERT_NE(eid,kNullUid);
    GXF_ASSERT_SUCCESS(GxfEntityRefCountInc(context, eid));
    ASSERT_EQ(GxfEntityRefCountDec(context,kNullUid),GXF_REF_COUNT_NEGATIVE);
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}
TEST(GxfEntityRefCountDec,null_eid_value)
{
    gxf_context_t context;
    gxf_uid_t eid = kNullUid;
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    const GxfEntityCreateInfo entity_create_info ={0};
    GXF_ASSERT_SUCCESS(GxfCreateEntity(context,&entity_create_info, &eid));
    ASSERT_NE(eid,kNullUid);
    GXF_ASSERT_SUCCESS(GxfEntityRefCountInc(context, eid));
    ASSERT_EQ(GxfEntityRefCountDec(context,-1),GXF_REF_COUNT_NEGATIVE);
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityRefCountDec,zero_eid_value)
{
    gxf_context_t context;
    gxf_uid_t eid = kNullUid;
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    const GxfEntityCreateInfo entity_create_info ={0};
    GXF_ASSERT_SUCCESS(GxfCreateEntity(context,&entity_create_info, &eid));
    ASSERT_NE(eid,kNullUid);
    GXF_ASSERT_SUCCESS(GxfEntityRefCountInc(context, eid));
    ASSERT_EQ(GxfEntityRefCountDec(context,0),GXF_REF_COUNT_NEGATIVE);
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}