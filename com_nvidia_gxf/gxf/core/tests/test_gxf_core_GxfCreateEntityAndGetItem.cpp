/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <iostream>
#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/core/entity_item.hpp"
#include "gxf/core/gxf.h"

TEST(Create_Entity, NULL_Value) {
  gxf_uid_t eid = kNullUid;
  const GxfEntityCreateInfo entity_create_info = {0};
  void* item_ptr = nullptr;
  ASSERT_EQ(GxfCreateEntityAndGetItem(NULL, &entity_create_info, &eid, &item_ptr),
            GXF_CONTEXT_INVALID);
  ASSERT_EQ(GxfContextDestroy(NULL), GXF_CONTEXT_INVALID);
}

TEST(Create_Entity, valid_context) {
  gxf_context_t context = kNullContext;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
  gxf_uid_t eid = kNullUid;
  const GxfEntityCreateInfo entity_create_info = {0};
  void* item_ptr = nullptr;
  ASSERT_EQ(GxfCreateEntityAndGetItem(context, &entity_create_info, &eid, &item_ptr), GXF_SUCCESS);
  ASSERT_NE(item_ptr, nullptr);
  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(Create_Entity, NULL_GxfEntityCreateInfo) {
  gxf_context_t context = kNullContext;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  gxf_uid_t eid = kNullUid;
  void* item_ptr = nullptr;
  GXF_ASSERT_EQ(GxfCreateEntityAndGetItem(context, nullptr, &eid, &item_ptr), GXF_ARGUMENT_NULL);
  ASSERT_EQ(item_ptr, nullptr);
}

TEST(Create_Entity, Invalid_GxfEntityCreateInfo) {
  gxf_context_t context = kNullContext;
  const char* InValid_Entity_Name = "__Entity1";
  const GxfEntityCreateInfo entity_create_info{InValid_Entity_Name};
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  gxf_uid_t eid = kNullUid;
  void* item_ptr = nullptr;
  ASSERT_EQ(GxfCreateEntityAndGetItem(context, &entity_create_info, &eid, &item_ptr),
            GXF_ARGUMENT_INVALID);
  ASSERT_EQ(item_ptr, nullptr);
  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_QUERY_NOT_FOUND);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(Create_Entity, Multiple_Entity_with_same_name) {
  gxf_context_t context = kNullContext;
  gxf_uid_t eid = kNullUid;
  gxf_uid_t eid1 = kNullUid;
  const char* Valid_Entity_Name = "Entity";
  const char* Valid_Entity_Name1 = "Entity";
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfEntityCreateInfo entity_create_info{Valid_Entity_Name};
  const GxfEntityCreateInfo entity_create_info1{Valid_Entity_Name1};
  void* item_ptr = nullptr;
  ASSERT_EQ(GxfCreateEntityAndGetItem(context, &entity_create_info, &eid, &item_ptr), GXF_SUCCESS);
  GXF_ASSERT_NE(item_ptr == nullptr, true);
  void* item_ptr1 = nullptr;
  ASSERT_EQ(GxfCreateEntityAndGetItem(context, &entity_create_info1, &eid1, &item_ptr1),
            GXF_ARGUMENT_INVALID);
  GXF_ASSERT_EQ(item_ptr1 == nullptr, true);
  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(Create_Entity, NULL_eid) {
  gxf_context_t context = kNullContext;
  const char* Valid_Entity_Name = "Entity";
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfEntityCreateInfo entity_create_info{Valid_Entity_Name};
  void* item_ptr = nullptr;
  GXF_ASSERT_EQ(GxfCreateEntityAndGetItem(context, &entity_create_info, NULL, &item_ptr),
                GXF_ARGUMENT_NULL);
  GXF_ASSERT_EQ(item_ptr == nullptr, true);
}

TEST(Create_Entity, NULL_ptr) {
  gxf_context_t context = kNullContext;
  const char* Valid_Entity_Name = "Entity";
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfEntityCreateInfo entity_create_info{Valid_Entity_Name};
  gxf_uid_t eid;
  GXF_ASSERT_EQ(GxfCreateEntityAndGetItem(context, &entity_create_info, &eid, nullptr),
                GXF_ARGUMENT_NULL);
}

TEST(Create_Entity, Non_NULL_ptr) {
  gxf_context_t context = kNullContext;
  const char* Valid_Entity_Name = "Entity";
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfEntityCreateInfo entity_create_info{Valid_Entity_Name};
  gxf_uid_t eid;
  void* item_ptr = reinterpret_cast<void*>(&eid);

  GXF_ASSERT_EQ(GxfCreateEntityAndGetItem(context, &entity_create_info, &eid, &item_ptr),
                GXF_ARGUMENT_INVALID);
}
