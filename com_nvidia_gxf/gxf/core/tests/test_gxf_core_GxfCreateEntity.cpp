/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/core/gxf.h"
#include <iostream>
#include "gtest/gtest.h"
#include "common/assert.hpp"

TEST(Create_Entity,NULL_Value)
{  gxf_uid_t eid = kNullUid;
   const GxfEntityCreateInfo entity_create_info = {0};
   ASSERT_EQ(GxfCreateEntity(NULL,&entity_create_info, &eid),GXF_CONTEXT_INVALID);
   ASSERT_EQ(GxfContextDestroy(NULL),GXF_CONTEXT_INVALID);
}

TEST(Create_Entity,valid_context)
{  gxf_context_t context = kNullContext;
   ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
   gxf_uid_t eid = kNullUid;
   const GxfEntityCreateInfo entity_create_info = {0};
   ASSERT_EQ(GxfCreateEntity(context,&entity_create_info, &eid),GXF_SUCCESS);
   ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
   ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(Create_Entity,NULL_GxfEntityCreateInfo)
{  gxf_context_t context = kNullContext;
   GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
   gxf_uid_t eid = kNullUid;
   GXF_ASSERT_EQ(GxfCreateEntity(context,nullptr, &eid),GXF_ARGUMENT_NULL);
}

TEST(Create_Entity,Invalid_GxfEntityCreateInfo)
{  gxf_context_t context = kNullContext;
   const char* InValid_Entity_Name="__Entity1";
   const GxfEntityCreateInfo entity_create_info{InValid_Entity_Name};
   GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
   gxf_uid_t eid = kNullUid;
   ASSERT_EQ(GxfCreateEntity(context,&entity_create_info, &eid),GXF_ARGUMENT_INVALID);
   ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_QUERY_NOT_FOUND);
   ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(Create_Entity,Multiple_Entity_with_same_name)
{  gxf_context_t context = kNullContext;
   gxf_uid_t eid = kNullUid;
   gxf_uid_t eid1 = kNullUid;
   const char* Valid_Entity_Name="Entity";
   const char* Valid_Entity_Name1="Entity";
   GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
   const GxfEntityCreateInfo entity_create_info{Valid_Entity_Name};
   const GxfEntityCreateInfo entity_create_info1{Valid_Entity_Name1};
   ASSERT_EQ(GxfCreateEntity(context,&entity_create_info, &eid),GXF_SUCCESS);
   ASSERT_EQ(GxfCreateEntity(context,&entity_create_info1, &eid1),GXF_ARGUMENT_INVALID);
   ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
   ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(Create_Entity,NULL_eid)
{  gxf_context_t context = kNullContext;
   const char* Valid_Entity_Name="Entity";
   GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
   const GxfEntityCreateInfo entity_create_info{Valid_Entity_Name};
   GXF_ASSERT_EQ(GxfCreateEntity(context,&entity_create_info,NULL),GXF_ARGUMENT_NULL);
}
