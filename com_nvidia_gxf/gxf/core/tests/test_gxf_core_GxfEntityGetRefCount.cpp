/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

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

TEST(GxfEntityGetRefCount, valid_context) {
  gxf_context_t context;
  gxf_uid_t eid = kNullUid;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);
  int64_t ref_count = -1;
  ASSERT_EQ(GxfEntityGetRefCount(context, eid, &ref_count), GXF_PARAMETER_NOT_FOUND);
  ASSERT_EQ(ref_count, -1);
  ASSERT_NE(eid, kNullUid);
  GXF_ASSERT_SUCCESS(GxfEntityRefCountInc(context, eid));
  ASSERT_EQ(GxfEntityGetRefCount(context, eid, &ref_count), GXF_SUCCESS);
  ASSERT_EQ(ref_count, 1);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(GxfEntityGetRefCount, null_context) {
  gxf_context_t context;
  gxf_uid_t eid = kNullUid;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfEntityCreateInfo entity_create_info = {0};
  GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));
  ASSERT_NE(eid, kNullUid);
  ASSERT_EQ(GxfEntityRefCountInc(NULL, eid), GXF_CONTEXT_INVALID);
  int64_t ref_count = 0;
  ASSERT_EQ(GxfEntityGetRefCount(NULL, eid, &ref_count), GXF_CONTEXT_INVALID);
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityGetRefCount, null_eid) {
  gxf_context_t context;
  gxf_uid_t eid = kNullUid;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfEntityCreateInfo entity_create_info = {0};
  GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));
  ASSERT_NE(eid, kNullUid);
  int64_t ref_count = 0;
  GXF_ASSERT_EQ(GxfEntityGetRefCount(context, kNullUid, &ref_count), GXF_PARAMETER_NOT_FOUND);
  GXF_ASSERT_EQ(ref_count, 0);
  GXF_ASSERT_SUCCESS(GxfEntityRefCountInc(context, kNullUid));
  GXF_ASSERT_SUCCESS(GxfEntityGetRefCount(context, kNullUid, &ref_count));
  GXF_ASSERT_EQ(ref_count, 1);
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityGetRefCount, zero_eid_value) {
  gxf_context_t context;
  gxf_uid_t eid = kNullUid;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfEntityCreateInfo entity_create_info = {0};
  GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));
  ASSERT_NE(eid, kNullUid);
  GXF_ASSERT_SUCCESS(GxfEntityRefCountInc(context, eid));
  int64_t ref_count = -2;
  GXF_ASSERT_SUCCESS(GxfEntityGetRefCount(context, eid, &ref_count));
  GXF_ASSERT_EQ(ref_count, 1);
  ASSERT_EQ(GxfEntityRefCountDec(context, 0), GXF_REF_COUNT_NEGATIVE);
  ref_count = -1;
  GXF_ASSERT_EQ(GxfEntityGetRefCount(context, 0, &ref_count), GXF_PARAMETER_NOT_FOUND);
  GXF_ASSERT_EQ(ref_count, -1);
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityGetRefCount, negative_eid) {
  gxf_context_t context;
  gxf_uid_t eid = kNullUid;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfEntityCreateInfo entity_create_info = {0};
  GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));
  ASSERT_NE(eid, kNullUid);
  GXF_ASSERT_SUCCESS(GxfEntityRefCountInc(context, -1));
  int64_t ref_count = -1;
  GXF_ASSERT_SUCCESS(GxfEntityGetRefCount(context, -1, &ref_count));
  GXF_ASSERT_EQ(ref_count, 1);
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}
