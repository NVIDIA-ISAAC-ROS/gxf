/*
Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/core/gxf.h"

#include "gtest/gtest.h"

TEST(Entity, Context) {
  gxf_context_t context = kNullContext;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(Entity, Create) {
  gxf_context_t context = kNullContext;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  gxf_uid_t eid = kNullUid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);
  ASSERT_NE(eid, kNullUid);
  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);

  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(Entity, Find) {
  gxf_context_t context = kNullContext;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  gxf_uid_t eid = kNullUid;
  const GxfEntityCreateInfo entity_create_info = {"foo", 0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);
  ASSERT_NE(eid, kNullUid);
  const char * entity_name = "UNKNOWN";
  ASSERT_EQ(GxfParameterGetStr(context, eid, kInternalNameParameterKey, &entity_name), GXF_SUCCESS);
  ASSERT_EQ(std::string("foo"), std::string(entity_name));

  gxf_uid_t other = kNullUid;
  ASSERT_EQ(GxfEntityFind(context, "foo", &other), GXF_SUCCESS);
  ASSERT_EQ(other, eid);

  gxf_uid_t other2 = kNullUid;
  ASSERT_EQ(GxfEntityFind(context, "foof", &other2), GXF_ENTITY_NOT_FOUND);
  ASSERT_EQ(other2, kNullUid);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);

  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(Entity, InvalidGxfEntityCreateInfo)
{
  gxf_context_t context = kNullContext;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  const char* entity_name="__Entity1";
  gxf_uid_t eid = kNullUid;
  const GxfEntityCreateInfo entity_create_info{entity_name};

  ASSERT_EQ(GxfCreateEntity(context,&entity_create_info, &eid), GXF_ARGUMENT_INVALID);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}
