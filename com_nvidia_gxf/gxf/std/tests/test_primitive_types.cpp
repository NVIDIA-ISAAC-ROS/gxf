/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "common/type_name.hpp"
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"

namespace nvidia {
namespace gxf {

namespace {

constexpr const char* kExtensions[] = {
  "gxf/std/libgxf_std.so",
};

}  // namespace

TEST(TestPrimitiveTypes, int8_t) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  const GxfLoadExtensionsInfo extension_info{kExtensions, 1, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &extension_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, TypenameAsString<int8_t>(), &tid), GXF_SUCCESS);
  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, nullptr, &cid), GXF_SUCCESS);
  int8_t* data;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, reinterpret_cast<void**>(&data)), GXF_SUCCESS);

  *data = 0x01;
  ASSERT_EQ(*data, 0x01);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(TestPrimitiveTypes, uint8_t) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  const GxfLoadExtensionsInfo extension_info{kExtensions, 1, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &extension_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, TypenameAsString<uint8_t>(), &tid), GXF_SUCCESS);
  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, nullptr, &cid), GXF_SUCCESS);
  uint8_t* data;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, reinterpret_cast<void**>(&data)), GXF_SUCCESS);

  *data = 0x01;
  ASSERT_EQ(*data, 0x01);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(TestPrimitiveTypes, int16_t) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  const GxfLoadExtensionsInfo extension_info{kExtensions, 1, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &extension_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, TypenameAsString<int16_t>(), &tid), GXF_SUCCESS);
  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, nullptr, &cid), GXF_SUCCESS);
  int16_t* data;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, reinterpret_cast<void**>(&data)), GXF_SUCCESS);

  *data = 0x0123;
  ASSERT_EQ(*data, 0x0123);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(TestPrimitiveTypes, uint16_t) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  const GxfLoadExtensionsInfo extension_info{kExtensions, 1, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &extension_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, TypenameAsString<uint16_t>(), &tid), GXF_SUCCESS);
  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, nullptr, &cid), GXF_SUCCESS);
  uint16_t* data;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, reinterpret_cast<void**>(&data)), GXF_SUCCESS);

  *data = 0x0123;
  ASSERT_EQ(*data, 0x0123);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(TestPrimitiveTypes, int32_t) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  const GxfLoadExtensionsInfo extension_info{kExtensions, 1, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &extension_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, TypenameAsString<int32_t>(), &tid), GXF_SUCCESS);
  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, nullptr, &cid), GXF_SUCCESS);
  int32_t* data;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, reinterpret_cast<void**>(&data)), GXF_SUCCESS);

  *data = 0x01234567;
  ASSERT_EQ(*data, 0x01234567);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(TestPrimitiveTypes, uint32_t) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  const GxfLoadExtensionsInfo extension_info{kExtensions, 1, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &extension_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, TypenameAsString<uint32_t>(), &tid), GXF_SUCCESS);
  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, nullptr, &cid), GXF_SUCCESS);
  uint32_t* data;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, reinterpret_cast<void**>(&data)), GXF_SUCCESS);

  *data = 0x01234567;
  ASSERT_EQ(*data, 0x01234567);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(TestPrimitiveTypes, int64_t) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  const GxfLoadExtensionsInfo extension_info{kExtensions, 1, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &extension_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, TypenameAsString<int64_t>(), &tid), GXF_SUCCESS);
  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, nullptr, &cid), GXF_SUCCESS);
  int64_t* data;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, reinterpret_cast<void**>(&data)), GXF_SUCCESS);

  *data = 0x0123456789ABCDEF;
  ASSERT_EQ(*data, 0x0123456789ABCDEF);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(TestPrimitiveTypes, uint64_t) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  const GxfLoadExtensionsInfo extension_info{kExtensions, 1, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &extension_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, TypenameAsString<uint64_t>(), &tid), GXF_SUCCESS);
  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, nullptr, &cid), GXF_SUCCESS);
  uint64_t* data;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, reinterpret_cast<void**>(&data)), GXF_SUCCESS);

  *data = 0x0123456789ABCDEF;
  ASSERT_EQ(*data, 0x0123456789ABCDEF);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(TestPrimitiveTypes, float) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  const GxfLoadExtensionsInfo extension_info{kExtensions, 1, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &extension_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, TypenameAsString<float>(), &tid), GXF_SUCCESS);
  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, nullptr, &cid), GXF_SUCCESS);
  float* data;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, reinterpret_cast<void**>(&data)), GXF_SUCCESS);

  *data = 3.14;
  ASSERT_FLOAT_EQ(*data, 3.14);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(TestPrimitiveTypes, double) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  const GxfLoadExtensionsInfo extension_info{kExtensions, 1, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &extension_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, TypenameAsString<double>(), &tid), GXF_SUCCESS);
  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, nullptr, &cid), GXF_SUCCESS);
  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);
  double* data;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, reinterpret_cast<void**>(&data)), GXF_SUCCESS);

  *data = 3.14159265359;
  ASSERT_DOUBLE_EQ(*data, 3.14159265359);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(TestPrimitiveTypes, bool) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  const GxfLoadExtensionsInfo extension_info{kExtensions, 1, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &extension_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, TypenameAsString<bool>(), &tid), GXF_SUCCESS);
  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, nullptr, &cid), GXF_SUCCESS);
  bool* data;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, reinterpret_cast<void**>(&data)), GXF_SUCCESS);

  *data = true;
  ASSERT_TRUE(*data);

  ASSERT_EQ(GxfEntityDestroy(context, eid), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

}  // namespace gxf
}  // namespace nvidia
