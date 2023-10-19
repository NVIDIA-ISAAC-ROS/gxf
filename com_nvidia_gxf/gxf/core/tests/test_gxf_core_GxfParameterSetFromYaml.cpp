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

#include <yaml-cpp/yaml.h>

#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/test/extensions/test_parameters.hpp"

namespace {
   constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";
}  // namespace


TEST(test_gxf_core_GxfParameterSetFromYaml,valid_testcase)
{
   gxf_context_t context = kNullContext;
   gxf_uid_t eid = kNullUid;
   gxf_tid_t tid = GxfTidNull();
   gxf_uid_t cid = kNullUid;

   const GxfEntityCreateInfo entity_create_info = {0};
   const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};

   GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
   GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));

   GXF_ASSERT_SUCCESS(GxfCreateEntity(context,&entity_create_info, &eid));
   GXF_ASSERT_SUCCESS(GxfComponentTypeId(context,"nvidia::gxf::BlockMemoryPool", &tid));
   ASSERT_NE(GxfTidNull(), tid);
   GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "test", &cid));
   ASSERT_NE(kNullUid,cid);

   // uint64_t parameter
   YAML::Node block_size_node{10};
   uint64_t block_size_value;
   GXF_ASSERT_SUCCESS(GxfParameterSetFromYamlNode(context,cid,"block_size",&block_size_node,""));
   GXF_ASSERT_SUCCESS(GxfParameterGetUInt64(context, cid,"block_size",&block_size_value));
   GXF_ASSERT_EQ(block_size_value,10);

   // int32_t parameter
   YAML::Node storage_type_node{1};
   int32_t storage_type_value;
   GXF_ASSERT_SUCCESS(GxfParameterSetFromYamlNode(context,cid,"storage_type",&storage_type_node,""));
   GXF_ASSERT_SUCCESS(GxfParameterGetInt32(context, cid,"storage_type",&storage_type_value));
   GXF_ASSERT_EQ(storage_type_value,1);
   GXF_ASSERT_EQ(GxfContextDestroy(context),GXF_SUCCESS);
}

TEST(test_gxf_core_GxfParameterSetFromYaml, various_types_testcase) {
  gxf_context_t context = kNullContext;
  gxf_uid_t eid = kNullUid;
  gxf_uid_t cid = kNullUid;

  const GxfEntityCreateInfo entity_create_info = {0};
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};

  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));

  gxf_tid_t allocator_tid;
  gxf_uid_t allocator_cid = kNullUid;
  gxf_uid_t allocator2_cid = kNullUid;
  GXF_ASSERT_SUCCESS(
      GxfComponentTypeId(context, "nvidia::gxf::UnboundedAllocator", &allocator_tid));
  ASSERT_NE(GxfTidNull(), allocator_tid);
  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, allocator_tid, "allocator", &allocator_cid));
  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, allocator_tid, "allocator2", &allocator2_cid));

  gxf_tid_t tid = GxfTidNull();
  GXF_ASSERT_SUCCESS(
      GxfComponentTypeId(context, "nvidia::gxf::test::TestGxfParameterSetFromYamlNode", &tid));
  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "test", &cid));

  void* component_ptr = nullptr;
  GxfComponentPointer(context, cid, tid, &component_ptr);
  GXF_ASSERT_TRUE(component_ptr != nullptr);

  nvidia::gxf::test::TestGxfParameterSetFromYamlNode* obj =
      static_cast<nvidia::gxf::test::TestGxfParameterSetFromYamlNode*>(component_ptr);

  // bool parameter
  YAML::Node bool_node{true};
  GXF_ASSERT_SUCCESS(GxfParameterSetFromYamlNode(context, cid, "bool", &bool_node, ""));
  GXF_ASSERT_EQ(obj->bool_.get(), true);
  // int8 parameter
  YAML::Node int8_node{static_cast<int8_t>(-35)};
  GXF_ASSERT_SUCCESS(GxfParameterSetFromYamlNode(context, cid, "int8", &int8_node, ""));
  GXF_ASSERT_EQ(obj->int8_.get(), -35);
  // int16 parameter
  YAML::Node int16_node{static_cast<int16_t>(-25532)};
  GXF_ASSERT_SUCCESS(GxfParameterSetFromYamlNode(context, cid, "int16", &int16_node, ""));
  GXF_ASSERT_EQ(obj->int16_.get(), -25532);
  // int32 parameter
  YAML::Node int32_node{static_cast<int32_t>(-3411232)};
  GXF_ASSERT_SUCCESS(GxfParameterSetFromYamlNode(context, cid, "int32", &int32_node, ""));
  GXF_ASSERT_EQ(obj->int32_.get(), -3411232);
  // int64 parameter
  YAML::Node int64_node{static_cast<int64_t>(-5434240006L)};
  GXF_ASSERT_SUCCESS(GxfParameterSetFromYamlNode(context, cid, "int64", &int64_node, ""));
  GXF_ASSERT_EQ(obj->int64_.get(), -5434240006L);
  // uint8 parameter
  // : uint8_t is not supported natively by yaml-cpp so push it as a uint32_t so that GXF can handle
  //   it.
  YAML::Node uint8_node{static_cast<uint32_t>(74)};
  GXF_ASSERT_SUCCESS(GxfParameterSetFromYamlNode(context, cid, "uint8", &uint8_node, ""));
  GXF_ASSERT_EQ(obj->uint8_.get(), static_cast<uint8_t>(74));
  // uint16 parameter
  YAML::Node uint16_node{static_cast<uint16_t>(13405)};
  GXF_ASSERT_SUCCESS(GxfParameterSetFromYamlNode(context, cid, "uint16", &uint16_node, ""));
  GXF_ASSERT_EQ(obj->uint16_.get(), 13405);
  // uint32 parameter
  YAML::Node uint32_node{static_cast<uint32_t>(3411232)};
  GXF_ASSERT_SUCCESS(GxfParameterSetFromYamlNode(context, cid, "uint32", &uint32_node, ""));
  GXF_ASSERT_EQ(obj->uint32_.get(), 3411232);
  // uint64 parameter
  YAML::Node uint64_node{static_cast<uint64_t>(5434240006L)};
  GXF_ASSERT_SUCCESS(GxfParameterSetFromYamlNode(context, cid, "uint64", &uint64_node, ""));
  GXF_ASSERT_EQ(obj->uint64_.get(), 5434240006L);
  // float parameter
  YAML::Node float_node{static_cast<float>(3.14f)};
  GXF_ASSERT_SUCCESS(GxfParameterSetFromYamlNode(context, cid, "float", &float_node, ""));
  GXF_ASSERT_EQ(obj->float_.get(), 3.14f);
  // double parameter
  YAML::Node double_node{static_cast<double>(3.1415926535)};
  GXF_ASSERT_SUCCESS(GxfParameterSetFromYamlNode(context, cid, "double", &double_node, ""));
  GXF_ASSERT_EQ(obj->double_.get(), 3.1415926535);
  // string parameter
  YAML::Node string_node{"my string"};
  GXF_ASSERT_SUCCESS(GxfParameterSetFromYamlNode(context, cid, "string", &string_node, ""));
  GXF_ASSERT_TRUE(obj->string_.get() == "my string");
  // handle parameter
  YAML::Node handle_node{"allocator"};
  GXF_ASSERT_SUCCESS(GxfParameterSetFromYamlNode(context, cid, "handle", &handle_node, ""));
  GXF_ASSERT_EQ(obj->handle_.get().cid(), allocator_cid);

  // vector_bool parameter
  YAML::Node vector_bool_node;
  vector_bool_node.push_back(true);
  vector_bool_node.push_back(false);
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_bool", &vector_bool_node, ""));
  GXF_ASSERT_EQ(obj->vector_bool_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_bool_.get().at(0), true);
  GXF_ASSERT_EQ(obj->vector_bool_.get().at(1), false);
  // vector_int8 parameter
  YAML::Node vector_int8_node;
  vector_int8_node.push_back(static_cast<int8_t>(-35));
  vector_int8_node.push_back(static_cast<int8_t>(54));
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_int8", &vector_int8_node, ""));
  GXF_ASSERT_EQ(obj->vector_int8_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_int8_.get().at(0), -35);
  GXF_ASSERT_EQ(obj->vector_int8_.get().at(1), 54);
  // vector_int16 parameter
  YAML::Node vector_int16_node;
  vector_int16_node.push_back(static_cast<int16_t>(-25532));
  vector_int16_node.push_back(static_cast<int16_t>(13405));
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_int16", &vector_int16_node, ""));
  GXF_ASSERT_EQ(obj->vector_int16_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_int16_.get().at(0), -25532);
  GXF_ASSERT_EQ(obj->vector_int16_.get().at(1), 13405);
  // vector_int32 parameter
  YAML::Node vector_int32_node;
  vector_int32_node.push_back(static_cast<int32_t>(-3411232));
  vector_int32_node.push_back(static_cast<int32_t>(3411232));
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_int32", &vector_int32_node, ""));
  GXF_ASSERT_EQ(obj->vector_int32_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_int32_.get().at(0), -3411232);
  GXF_ASSERT_EQ(obj->vector_int32_.get().at(1), 3411232);
  // vector_int64 parameter
  YAML::Node vector_int64_node;
  vector_int64_node.push_back(static_cast<int64_t>(-5434240006L));
  vector_int64_node.push_back(static_cast<int64_t>(5434240006L));
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_int64", &vector_int64_node, ""));
  GXF_ASSERT_EQ(obj->vector_int64_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_int64_.get().at(0), -5434240006L);
  GXF_ASSERT_EQ(obj->vector_int64_.get().at(1), 5434240006L);
  // vector_uint8 parameter
  // : uint8_t is not supported natively by yaml-cpp so push it as a uint32_t so that GXF can handle
  //   it.
  YAML::Node vector_uint8_node;
  vector_uint8_node.push_back(static_cast<uint32_t>(35));
  vector_uint8_node.push_back(static_cast<uint32_t>(54));
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_uint8", &vector_uint8_node, ""));
  GXF_ASSERT_EQ(obj->vector_uint8_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_uint8_.get().at(0), 35);
  GXF_ASSERT_EQ(obj->vector_uint8_.get().at(1), 54);
  // vector_uint16 parameter
  YAML::Node vector_uint16_node;
  vector_uint16_node.push_back(static_cast<uint16_t>(25532));
  vector_uint16_node.push_back(static_cast<uint16_t>(13405));
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_uint16", &vector_uint16_node, ""));
  GXF_ASSERT_EQ(obj->vector_uint16_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_uint16_.get().at(0), 25532);
  GXF_ASSERT_EQ(obj->vector_uint16_.get().at(1), 13405);
  // vector_uint32 parameter
  YAML::Node vector_uint32_node;
  vector_uint32_node.push_back(static_cast<uint32_t>(6411232));
  vector_uint32_node.push_back(static_cast<uint32_t>(3411232));
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_uint32", &vector_uint32_node, ""));
  GXF_ASSERT_EQ(obj->vector_uint32_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_uint32_.get().at(0), 6411232);
  GXF_ASSERT_EQ(obj->vector_uint32_.get().at(1), 3411232);
  // vector_uint64 parameter
  YAML::Node vector_uint64_node;
  vector_uint64_node.push_back(static_cast<uint64_t>(0L));
  vector_uint64_node.push_back(static_cast<uint64_t>(5434240006L));
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_uint64", &vector_uint64_node, ""));
  GXF_ASSERT_EQ(obj->vector_uint64_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_uint64_.get().at(0), 0L);
  GXF_ASSERT_EQ(obj->vector_uint64_.get().at(1), 5434240006L);
  // vector_float parameter
  YAML::Node vector_float_node;
  vector_float_node.push_back(static_cast<float>(3.14f));
  vector_float_node.push_back(static_cast<float>(2.718f));
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_float", &vector_float_node, ""));
  GXF_ASSERT_EQ(obj->vector_float_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_float_.get().at(0), 3.14f);
  GXF_ASSERT_EQ(obj->vector_float_.get().at(1), 2.718f);
  // vector_double parameter
  YAML::Node vector_double_node;
  vector_double_node.push_back(static_cast<double>(3.1415926535));
  vector_double_node.push_back(static_cast<double>(-0.71856753));
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_double", &vector_double_node, ""));
  GXF_ASSERT_EQ(obj->vector_double_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_double_.get().at(0), 3.1415926535);
  GXF_ASSERT_EQ(obj->vector_double_.get().at(1), -0.71856753);
  // vector_handle parameter
  YAML::Node vector_handle_node;
  vector_handle_node.push_back("allocator");
  vector_handle_node.push_back("allocator2");
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_handle", &vector_handle_node, ""));
  GXF_ASSERT_EQ(obj->vector_handle_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_handle_.get().at(0).cid(), allocator_cid);
  GXF_ASSERT_EQ(obj->vector_handle_.get().at(1).cid(), allocator2_cid);
  // vector_string parameter
  YAML::Node vector_string_node;
  vector_string_node.push_back("string1");
  vector_string_node.push_back("string2");
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_string", &vector_string_node, ""));
  GXF_ASSERT_EQ(obj->vector_string_.get().size(), 2);
  GXF_ASSERT_TRUE(obj->vector_string_.get().at(0) == "string1");
  GXF_ASSERT_TRUE(obj->vector_string_.get().at(1) == "string2");

  // vector_2d_bool parameter
  YAML::Node vector_2d_bool_node = YAML::Load("[[true, false], [false, true]]");
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_2d_bool", &vector_2d_bool_node, ""));
  GXF_ASSERT_EQ(obj->vector_2d_bool_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_bool_.get().at(0).size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_bool_.get().at(0).at(0), true);
  GXF_ASSERT_EQ(obj->vector_2d_bool_.get().at(0).at(1), false);
  GXF_ASSERT_EQ(obj->vector_2d_bool_.get().at(1).size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_bool_.get().at(1).at(0), false);
  GXF_ASSERT_EQ(obj->vector_2d_bool_.get().at(1).at(1), true);
  // vector_2d_int8 parameter
  YAML::Node vector_2d_int8_node = YAML::Load("[]");
  YAML::Node vector_2d_int8_node_1 = YAML::Load("[]");
  YAML::Node vector_2d_int8_node_2 = YAML::Load("[]");
  vector_2d_int8_node_1.push_back(static_cast<int8_t>(-1));
  vector_2d_int8_node_1.push_back(static_cast<int8_t>(2));
  vector_2d_int8_node_2.push_back(static_cast<int8_t>(3));
  vector_2d_int8_node_2.push_back(static_cast<int8_t>(-4));
  vector_2d_int8_node.push_back(vector_2d_int8_node_1);
  vector_2d_int8_node.push_back(vector_2d_int8_node_2);
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_2d_int8", &vector_2d_int8_node, ""));
  GXF_ASSERT_EQ(obj->vector_2d_int8_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_int8_.get().at(0).size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_int8_.get().at(0).at(0), -1);
  GXF_ASSERT_EQ(obj->vector_2d_int8_.get().at(0).at(1), 2);
  GXF_ASSERT_EQ(obj->vector_2d_int8_.get().at(1).size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_int8_.get().at(1).at(0), 3);
  GXF_ASSERT_EQ(obj->vector_2d_int8_.get().at(1).at(1), -4);
  // vector_2d_int16 parameter
  YAML::Node vector_2d_int16_node = YAML::Load("[]");
  YAML::Node vector_2d_int16_node_1 = YAML::Load("[]");
  YAML::Node vector_2d_int16_node_2 = YAML::Load("[]");
  vector_2d_int16_node_1.push_back(static_cast<int16_t>(-15335));
  vector_2d_int16_node_1.push_back(static_cast<int16_t>(2335));
  vector_2d_int16_node_2.push_back(static_cast<int16_t>(15366));
  vector_2d_int16_node_2.push_back(static_cast<int16_t>(-4332));
  vector_2d_int16_node.push_back(vector_2d_int16_node_1);
  vector_2d_int16_node.push_back(vector_2d_int16_node_2);
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_2d_int16", &vector_2d_int16_node, ""));
  GXF_ASSERT_EQ(obj->vector_2d_int16_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_int16_.get().at(0).size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_int16_.get().at(0).at(0), -15335);
  GXF_ASSERT_EQ(obj->vector_2d_int16_.get().at(0).at(1), 2335);
  GXF_ASSERT_EQ(obj->vector_2d_int16_.get().at(1).size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_int16_.get().at(1).at(0), 15366);
  GXF_ASSERT_EQ(obj->vector_2d_int16_.get().at(1).at(1), -4332);
  // vector_2d_int32 parameter
  YAML::Node vector_2d_int32_node = YAML::Load("[]");
  YAML::Node vector_2d_int32_node_1 = YAML::Load("[]");
  YAML::Node vector_2d_int32_node_2 = YAML::Load("[]");
  vector_2d_int32_node_1.push_back(static_cast<int32_t>(-1645335));
  vector_2d_int32_node_1.push_back(static_cast<int32_t>(2356435));
  vector_2d_int32_node_2.push_back(static_cast<int32_t>(6615366));
  vector_2d_int32_node_2.push_back(static_cast<int32_t>(-23454332));
  vector_2d_int32_node.push_back(vector_2d_int32_node_1);
  vector_2d_int32_node.push_back(vector_2d_int32_node_2);
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_2d_int32", &vector_2d_int32_node, ""));
  GXF_ASSERT_EQ(obj->vector_2d_int32_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_int32_.get().at(0).size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_int32_.get().at(0).at(0), -1645335);
  GXF_ASSERT_EQ(obj->vector_2d_int32_.get().at(0).at(1), 2356435);
  GXF_ASSERT_EQ(obj->vector_2d_int32_.get().at(1).size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_int32_.get().at(1).at(0), 6615366);
  GXF_ASSERT_EQ(obj->vector_2d_int32_.get().at(1).at(1), -23454332);
  // vector_2d_int64 parameter
  YAML::Node vector_2d_int64_node = YAML::Load("[]");
  YAML::Node vector_2d_int64_node_1 = YAML::Load("[]");
  YAML::Node vector_2d_int64_node_2 = YAML::Load("[]");
  vector_2d_int64_node_1.push_back(static_cast<int64_t>(-2346665335));
  vector_2d_int64_node_1.push_back(static_cast<int64_t>(75356435));
  vector_2d_int64_node_2.push_back(static_cast<int64_t>(-345615366));
  vector_2d_int64_node_2.push_back(static_cast<int64_t>(56623434442));
  vector_2d_int64_node.push_back(vector_2d_int64_node_1);
  vector_2d_int64_node.push_back(vector_2d_int64_node_2);
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_2d_int64", &vector_2d_int64_node, ""));
  GXF_ASSERT_EQ(obj->vector_2d_int64_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_int64_.get().at(0).size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_int64_.get().at(0).at(0), -2346665335);
  GXF_ASSERT_EQ(obj->vector_2d_int64_.get().at(0).at(1), 75356435);
  GXF_ASSERT_EQ(obj->vector_2d_int64_.get().at(1).size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_int64_.get().at(1).at(0), -345615366);
  GXF_ASSERT_EQ(obj->vector_2d_int64_.get().at(1).at(1), 56623434442);
  // vector_2d_uint8 parameter
  // : uint8_t is not supported natively by yaml-cpp so push it as a uint32_t so that GXF can handle
  //   it.
  YAML::Node vector_2d_uint8_node = YAML::Load("[]");
  YAML::Node vector_2d_uint8_node_1 = YAML::Load("[]");
  YAML::Node vector_2d_uint8_node_2 = YAML::Load("[]");
  vector_2d_uint8_node_1.push_back(static_cast<uint32_t>(255));
  vector_2d_uint8_node_1.push_back(static_cast<uint32_t>(0));
  vector_2d_uint8_node_2.push_back(static_cast<uint32_t>(0));
  vector_2d_uint8_node_2.push_back(static_cast<uint32_t>(255));
  vector_2d_uint8_node.push_back(vector_2d_uint8_node_1);
  vector_2d_uint8_node.push_back(vector_2d_uint8_node_2);
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_2d_uint8", &vector_2d_uint8_node, ""));
  GXF_ASSERT_EQ(obj->vector_2d_uint8_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_uint8_.get().at(0).size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_uint8_.get().at(0).at(0), 255);
  GXF_ASSERT_EQ(obj->vector_2d_uint8_.get().at(0).at(1), 0);
  GXF_ASSERT_EQ(obj->vector_2d_uint8_.get().at(1).size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_uint8_.get().at(1).at(0), 0);
  GXF_ASSERT_EQ(obj->vector_2d_uint8_.get().at(1).at(1), 255);
  // vector_2d_uint16 parameter
  YAML::Node vector_2d_uint16_node = YAML::Load("[]");
  YAML::Node vector_2d_uint16_node_1 = YAML::Load("[]");
  YAML::Node vector_2d_uint16_node_2 = YAML::Load("[]");
  vector_2d_uint16_node_1.push_back(static_cast<uint16_t>(65535));
  vector_2d_uint16_node_1.push_back(static_cast<uint16_t>(0));
  vector_2d_uint16_node_2.push_back(static_cast<uint16_t>(0));
  vector_2d_uint16_node_2.push_back(static_cast<uint16_t>(65535));
  vector_2d_uint16_node.push_back(vector_2d_uint16_node_1);
  vector_2d_uint16_node.push_back(vector_2d_uint16_node_2);
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_2d_uint16", &vector_2d_uint16_node, ""));
  GXF_ASSERT_EQ(obj->vector_2d_uint16_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_uint16_.get().at(0).size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_uint16_.get().at(0).at(0), 65535);
  GXF_ASSERT_EQ(obj->vector_2d_uint16_.get().at(0).at(1), 0);
  GXF_ASSERT_EQ(obj->vector_2d_uint16_.get().at(1).size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_uint16_.get().at(1).at(0), 0);
  GXF_ASSERT_EQ(obj->vector_2d_uint16_.get().at(1).at(1), 65535);
  // vector_2d_uint32 parameter
  YAML::Node vector_2d_uint32_node = YAML::Load("[]");
  YAML::Node vector_2d_uint32_node_1 = YAML::Load("[]");
  YAML::Node vector_2d_uint32_node_2 = YAML::Load("[]");
  vector_2d_uint32_node_1.push_back(static_cast<uint32_t>(4294967295));
  vector_2d_uint32_node_1.push_back(static_cast<uint32_t>(0));
  vector_2d_uint32_node_2.push_back(static_cast<uint32_t>(0));
  vector_2d_uint32_node_2.push_back(static_cast<uint32_t>(4294967295));
  vector_2d_uint32_node.push_back(vector_2d_uint32_node_1);
  vector_2d_uint32_node.push_back(vector_2d_uint32_node_2);
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_2d_uint32", &vector_2d_uint32_node, ""));
  GXF_ASSERT_EQ(obj->vector_2d_uint32_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_uint32_.get().at(0).size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_uint32_.get().at(0).at(0), 4294967295);
  GXF_ASSERT_EQ(obj->vector_2d_uint32_.get().at(0).at(1), 0);
  GXF_ASSERT_EQ(obj->vector_2d_uint32_.get().at(1).size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_uint32_.get().at(1).at(0), 0);
  GXF_ASSERT_EQ(obj->vector_2d_uint32_.get().at(1).at(1), 4294967295);
  // vector_2d_uint64 parameter
  YAML::Node vector_2d_uint64_node = YAML::Load("[]");
  YAML::Node vector_2d_uint64_node_1 = YAML::Load("[]");
  YAML::Node vector_2d_uint64_node_2 = YAML::Load("[]");
  vector_2d_uint64_node_1.push_back(static_cast<uint64_t>(18446744073709551615UL));
  vector_2d_uint64_node_1.push_back(static_cast<uint64_t>(0));
  vector_2d_uint64_node_2.push_back(static_cast<uint64_t>(0));
  vector_2d_uint64_node_2.push_back(static_cast<uint64_t>(18446744073709551615UL));
  vector_2d_uint64_node.push_back(vector_2d_uint64_node_1);
  vector_2d_uint64_node.push_back(vector_2d_uint64_node_2);
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_2d_uint64", &vector_2d_uint64_node, ""));
  GXF_ASSERT_EQ(obj->vector_2d_uint64_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_uint64_.get().at(0).size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_uint64_.get().at(0).at(0), 18446744073709551615UL);
  GXF_ASSERT_EQ(obj->vector_2d_uint64_.get().at(0).at(1), 0);
  GXF_ASSERT_EQ(obj->vector_2d_uint64_.get().at(1).size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_uint64_.get().at(1).at(0), 0);
  GXF_ASSERT_EQ(obj->vector_2d_uint64_.get().at(1).at(1), 18446744073709551615UL);
  // vector_2d_float parameter
  YAML::Node vector_2d_float_node = YAML::Load("[]");
  YAML::Node vector_2d_float_node_1 = YAML::Load("[]");
  YAML::Node vector_2d_float_node_2 = YAML::Load("[]");
  vector_2d_float_node_1.push_back(static_cast<float>(1.0));
  vector_2d_float_node_1.push_back(static_cast<float>(0.0));
  vector_2d_float_node_2.push_back(static_cast<float>(0.0));
  vector_2d_float_node_2.push_back(static_cast<float>(1.0));
  vector_2d_float_node.push_back(vector_2d_float_node_1);
  vector_2d_float_node.push_back(vector_2d_float_node_2);
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_2d_float", &vector_2d_float_node, ""));
  GXF_ASSERT_EQ(obj->vector_2d_float_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_float_.get().at(0).size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_float_.get().at(0).at(0), 1.0);
  GXF_ASSERT_EQ(obj->vector_2d_float_.get().at(0).at(1), 0.0);
  GXF_ASSERT_EQ(obj->vector_2d_float_.get().at(1).size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_float_.get().at(1).at(0), 0.0);
  GXF_ASSERT_EQ(obj->vector_2d_float_.get().at(1).at(1), 1.0);
  // vector_2d_double parameter
  YAML::Node vector_2d_double_node = YAML::Load("[]");
  YAML::Node vector_2d_double_node_1 = YAML::Load("[]");
  YAML::Node vector_2d_double_node_2 = YAML::Load("[]");
  vector_2d_double_node_1.push_back(static_cast<double>(1.0));
  vector_2d_double_node_1.push_back(static_cast<double>(0.0));
  vector_2d_double_node_2.push_back(static_cast<double>(0.0));
  vector_2d_double_node_2.push_back(static_cast<double>(1.0));
  vector_2d_double_node.push_back(vector_2d_double_node_1);
  vector_2d_double_node.push_back(vector_2d_double_node_2);
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_2d_double", &vector_2d_double_node, ""));
  GXF_ASSERT_EQ(obj->vector_2d_double_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_double_.get().at(0).size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_double_.get().at(0).at(0), 1.0);
  GXF_ASSERT_EQ(obj->vector_2d_double_.get().at(0).at(1), 0.0);
  GXF_ASSERT_EQ(obj->vector_2d_double_.get().at(1).size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_double_.get().at(1).at(0), 0.0);
  GXF_ASSERT_EQ(obj->vector_2d_double_.get().at(1).at(1), 1.0);
  // vector_2d_string parameter
  YAML::Node vector_2d_string_node = YAML::Load("[]");
  YAML::Node vector_2d_string_node_1 = YAML::Load("[]");
  YAML::Node vector_2d_string_node_2 = YAML::Load("[]");
  vector_2d_string_node_1.push_back("string1");
  vector_2d_string_node_1.push_back("string2");
  vector_2d_string_node_2.push_back("string3");
  vector_2d_string_node_2.push_back("string4");
  vector_2d_string_node.push_back(vector_2d_string_node_1);
  vector_2d_string_node.push_back(vector_2d_string_node_2);
  GXF_ASSERT_SUCCESS(
      GxfParameterSetFromYamlNode(context, cid, "vector_2d_string", &vector_2d_string_node, ""));
  GXF_ASSERT_EQ(obj->vector_2d_string_.get().size(), 2);
  GXF_ASSERT_EQ(obj->vector_2d_string_.get().at(0).size(), 2);
  GXF_ASSERT_TRUE(obj->vector_2d_string_.get().at(0).at(0) == "string1");
  GXF_ASSERT_TRUE(obj->vector_2d_string_.get().at(0).at(1) == "string2");
  GXF_ASSERT_EQ(obj->vector_2d_string_.get().at(1).size(), 2);
  GXF_ASSERT_TRUE(obj->vector_2d_string_.get().at(1).at(0) == "string3");
  GXF_ASSERT_TRUE(obj->vector_2d_string_.get().at(1).at(1) == "string4");

  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}
