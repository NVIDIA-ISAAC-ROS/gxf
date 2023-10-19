/*
Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.

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
#include "gxf/core/handle.hpp"
#include "gxf/std/default_extension.hpp"
#include "gxf/std/double_buffer_receiver.hpp"

#include "test_load_extension.hpp"

namespace {

constexpr const char* kStdExtensionFilename = "gxf/std/libgxf_std.so";
constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

}  // namespace

TEST(Entity, LoadExtension) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));

  constexpr const char* invalid_manifest_filename = "gxf/gxe/FileNotAvaible.yaml";
  const GxfLoadExtensionsInfo info_1{nullptr, 0, &invalid_manifest_filename, 1, nullptr};
  GXF_ASSERT_EQ(GxfLoadExtensions(context, &info_1), GXF_FAILURE);

  const GxfLoadExtensionsInfo info_2{&kStdExtensionFilename, 1, nullptr, 0, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info_2));

  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

/// Test that a custom extension with a codelet can be loaded by GxfLoadExtensionFromPointer().
TEST(Entity, LoadExtensionFromPointer) {
  // Create Context
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));

  // Load default extensions
  const GxfLoadExtensionsInfo info{&kStdExtensionFilename, 1, nullptr, 0, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));

  // GXF_EXT_FACTORY_BEGIN()
  auto factory = std::make_unique<nvidia::gxf::DefaultExtension>();
  GXF_ASSERT(factory, "Failed to create factory");

  // GXF_EXT_FACTORY_SET_INFO(0x51188a5b4bac4fb9,
  //                          0xa68012d305ae799c,
  //                          "GXF Test GxfLoadExtensionFromPointer",
  //                          "Test extension to demonstrate the use of
  //                          GxfLoadExtensionFromPointer",
  //                          "NVIDIA", "1.0.0", "NVIDIA");
  GXF_ASSERT(
      factory->setInfo({(0x51188a5b4bac4fb9), (0xa68012d305ae799c)},
                       "GXF Test GxfLoadExtensionFromPointer",
                       "Test extension to demonstrate the use of GxfLoadExtensionFromPointer",
                       "NVIDIA", "1.0.0", "NVIDIA"),
      "Failed to set extension info");

  // GXF_EXT_FACTORY_ADD(0x95858651ea2a4a34,
  //                     0xae670db3bec4cbbf,
  //                     nvidia::gxf::test::LoadExtensionFromPointerTest,
  //                     nvidia::gxf::Codelet,
  //                     "Test codelet to demonstrate the use of GxfLoadExtensionFromPointer");
  GXF_ASSERT(
    (factory->add<nvidia::gxf::test::LoadExtensionFromPointerTest, nvidia::gxf::Codelet>(
        {(0x95858651ea2a4a34), (0xae670db3bec4cbbf)},
        "Test codelet to demonstrate the use of GxfLoadExtensionFromPointer")),
    "Failed to add codelet");

  // Check if the factory is valid
  GXF_ASSERT(
    factory->checkInfo(),
    "Failed to check extension info");

  // GXF_EXT_FACTORY_END()
  nvidia::gxf::Extension* extension = factory.release();

  // Call GxfLoadExtensionFromPointer()
  GxfLoadExtensionFromPointer(context, extension);

  // Create Entity
  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));

  // Create Codelet
  gxf_tid_t codelet_tid;
  gxf_uid_t codelet_cid;
  GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::test::LoadExtensionFromPointerTest", &codelet_tid));
  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, codelet_tid, "test_codelet", &codelet_cid));

  // Set Codelet Parameters
  GXF_ASSERT_SUCCESS(GxfParameterSetInt32(context, codelet_cid, "value", static_cast<int32_t>(10)));

  // Validate Codelet
  void* codelet_ptr;
  GXF_ASSERT_SUCCESS(GxfComponentPointer(context, codelet_cid, codelet_tid, &codelet_ptr));
  nvidia::gxf::test::LoadExtensionFromPointerTest* codelet = static_cast<nvidia::gxf::test::LoadExtensionFromPointerTest*>(codelet_ptr);
  GXF_ASSERT(codelet->value() == 10, "Failed to set codelet parameter");

  int32_t value = 0;
  codelet->value(15);
  GxfParameterGetInt32(context, codelet_cid, "value", &value);
  GXF_ASSERT(value == 15, "Failed to get codelet parameter");

  // Destroy Context
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
  delete extension;
}

TEST(Entity, GxfRun) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));

  const GxfLoadExtensionsInfo info{&kStdExtensionFilename, 1, nullptr, 0, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));

  gxf_uid_t eid1;
  const GxfEntityCreateInfo entity_create_info = {0};
  GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid1));

  gxf_uid_t eid2 = kNullUid;
  GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid2));

  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRun(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));

  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, RunMultipleGraphs) {
constexpr const char* kExtensions[] = {
    "gxf/std/libgxf_std.so",
    "gxf/sample/libgxf_sample.so",
    "gxf/serialization/libgxf_serialization.so",
    "gxf/network/libgxf_network.so",
    "gxf/test/extensions/libgxf_test.so",
  };
  const GxfLoadExtensionsInfo load_extension_info{kExtensions, 5, nullptr, 0, nullptr};
  gxf_context_t context1;
  gxf_context_t shared_context;

  GXF_ASSERT_SUCCESS(GxfContextCreate(&context1));
  GXF_ASSERT_SUCCESS(GxfGetSharedContext(context1, &shared_context));

  gxf_context_t context2;
  GXF_ASSERT_SUCCESS(GxfContextCreate1(shared_context, &context2));

  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context1, &load_extension_info));

  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context1, "gxf/test/apps/test_ping.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFileExtended(context2, "gxf/test/apps/test_ping.yaml", "shared_ctx"));

  GXF_ASSERT_SUCCESS(GxfGraphActivate(context1));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context2));

  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context1));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context2));

  GXF_ASSERT_SUCCESS(GxfGraphWait(context1));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context2));

  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context1));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context2));

  GXF_ASSERT_SUCCESS(GxfContextDestroy(context2));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context1));
}

TEST(Entity, EntityRefCount) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));

  gxf_uid_t eid = kNullUid;
  const GxfEntityCreateInfo entity_create_info = {0};
  GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));

  GXF_ASSERT_EQ(GxfEntityRefCountDec(context, eid), GXF_REF_COUNT_NEGATIVE);
  GXF_ASSERT_SUCCESS(GxfEntityRefCountInc(context, eid));
  GXF_ASSERT_SUCCESS(GxfEntityRefCountInc(context, eid));
  GXF_ASSERT_SUCCESS(GxfEntityRefCountDec(context, eid));

  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, GxfEntityActivate) {
  // Test to verify entities are not activated if all the mandatory parameters
  // of the components are not set
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  GXF_ASSERT_SUCCESS(GxfLoadExtension(context, "gxf/std/libgxf_std.so"));
  GXF_ASSERT_SUCCESS(GxfLoadExtension(context, "gxf/test/extensions/libgxf_test.so"));
  gxf_uid_t eid = kNullUid;
  GXF_ASSERT_SUCCESS(GxfEntityCreate(context, &eid));
  gxf_tid_t rpi_tid{0xe9234c1ad5f8445c, 0xae9118bcda197032};  // RegisterParameterInterfaceTest
  gxf_uid_t rpi_uid = kNullUid;
  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, rpi_tid, "rpi", &rpi_uid));
  GXF_ASSERT_EQ(GxfEntityActivate(context, eid), GXF_PARAMETER_MANDATORY_NOT_SET);
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, ComponentFind) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));

  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));

  gxf_tid_t tid;
  GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::BlockMemoryPool", &tid));
  gxf_uid_t cid;
  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "test", &cid));
  int32_t offset;
  GXF_ASSERT_EQ(GxfComponentFind(context, eid, tid, "test1", &offset, &cid),
                GXF_ENTITY_COMPONENT_NOT_FOUND);

  GXF_ASSERT_SUCCESS(GxfEntityDestroy(context, eid));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, ComponentFindAll) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));

  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));

  gxf_tid_t tid;
  GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::BlockMemoryPool", &tid));
  gxf_uid_t cid;
  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "test1", &cid));
  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "test2", &cid));

  // Capacity limited by array size
  gxf_uid_t* cids = new gxf_uid_t[1];
  uint64_t cids_count = 1;
  gxf_result_t find_all_result = GxfComponentFindAll(context, eid, &cids_count, cids);
  GXF_ASSERT_EQ(find_all_result, GXF_QUERY_NOT_ENOUGH_CAPACITY);
  // Returned number of components should be more than the capacity
  GXF_ASSERT_EQ(cids_count, 2);
  delete[] cids;

  // Capacity not limited by array size
  cids = new gxf_uid_t[32];
  cids_count = 32;
  GXF_ASSERT_SUCCESS(GxfComponentFindAll(context, eid, &cids_count, cids));
  GXF_ASSERT_EQ(cids_count, 2);

  // Loop through to ensure cids are all nonzero and to check that the memory wasn't corrupted
  for (uint64_t i = 0; i < cids_count; i++) {
    GXF_ASSERT_NE(cids[i], 0);
  }
  delete[] cids;

  GXF_ASSERT_SUCCESS(GxfEntityDestroy(context, eid));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, GxfRegisterComponent) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  gxf_tid_t tid;
  GXF_ASSERT_SUCCESS(GxfRegisterComponent(context, tid, "nvidia::gxf::NewComponent", ""));
  gxf_tid_t tid1;
  GXF_ASSERT_EQ(GxfRegisterComponent(context, tid1, "nvidia::gxf::Component", ""),
                GXF_FACTORY_DUPLICATE_TID);
  GXF_ASSERT_EQ(
      GxfRegisterComponent(context, tid, "nvidia::gxf::NewComponent2", "Not_Registered_Base"),
      GXF_FACTORY_UNKNOWN_CLASS_NAME);

  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, GxfGraphParseString) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  GXF_ASSERT_SUCCESS(GxfGraphParseString(context, "name: dummy generator"));
  GXF_ASSERT_EQ(GxfGraphParseString(context, "dummy generator"), GXF_INVALID_DATA_FORMAT);

  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, GxfGraphSaveToFile) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));

  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {"test entity", 0};
  GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));

  gxf_tid_t tid;
  GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::BlockMemoryPool", &tid));
  gxf_uid_t cid;
  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "test comp 1", &cid));
  GXF_ASSERT_SUCCESS(GxfParameterSetInt32(context, cid, "storage_type", 0));
  GXF_ASSERT_SUCCESS(GxfParameterSetUInt64(context, cid, "block_size", 96));
  GXF_ASSERT_SUCCESS(GxfParameterSetUInt64(context, cid, "num_blocks", 3));

  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "test comp 2", &cid));

  // Export the entity/component graph to YAML
  GXF_ASSERT_SUCCESS(GxfGraphSaveToFile(context, "gxf/test/unit/test_save_to_file.yaml"));

  // Test loading the exported YAML graph
  gxf_context_t context_load;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context_load));

  const GxfLoadExtensionsInfo info_2{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context_load, &info_2));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context_load, "gxf/test/unit/test_save_to_file.yaml"));

  // Compare the exported and loaded graphs
  gxf_uid_t eid_load;
  GXF_ASSERT_SUCCESS(GxfEntityFind(context_load, "test entity", &eid_load));

  gxf_tid_t tid_load;
  GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::BlockMemoryPool", &tid_load));

  int offset = 0;
  gxf_uid_t cid_load;
  GXF_ASSERT_SUCCESS(GxfComponentFind(
    context_load, eid_load, tid_load, "test comp 1", &offset, &cid_load));

  // Check parameter values
  int32_t storage_type_value;
  uint64_t block_size_value;
  uint64_t num_blocks_value;
  GXF_ASSERT_SUCCESS(GxfParameterGetInt32(
    context_load, cid_load, "storage_type", &storage_type_value));
  GXF_ASSERT_SUCCESS(GxfParameterGetUInt64(
    context_load, cid_load, "block_size", &block_size_value));
  GXF_ASSERT_SUCCESS(GxfParameterGetUInt64(
    context_load, cid_load, "num_blocks", &num_blocks_value));

  GXF_ASSERT_EQ(storage_type_value, 0);
  GXF_ASSERT_EQ(block_size_value, 96);
  GXF_ASSERT_EQ(num_blocks_value, 3);

  GXF_ASSERT_SUCCESS(GxfComponentFind(
    context_load, eid_load, tid_load, "test comp 2", &offset, &cid_load));

  GXF_ASSERT_EQ(std::remove("gxf/test/unit/test_save_to_file.yaml"), 0);
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context_load));
}

TEST(Entity, ComponentEntity) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));

  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));

  gxf_tid_t tid;
  GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::BlockMemoryPool", &tid));
  gxf_uid_t cid;
  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "test", &cid));
  gxf_uid_t eid1;
  GXF_ASSERT_SUCCESS(GxfComponentEntity(context, cid, &eid1));
  GXF_ASSERT_EQ(GxfComponentEntity(context, eid, &eid1), GXF_ENTITY_NOT_FOUND);

  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, FindAll) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));

  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));

  gxf_uid_t single_entity;
  uint64_t entities_count = 0;
  GXF_ASSERT_SUCCESS(GxfEntityFindAll(context, &entities_count, &single_entity));
  // No entities should have been loaded yet
  GXF_ASSERT_EQ(entities_count, 0);

  // Create a single entity
  const GxfEntityCreateInfo entity_create_info = {0};
  GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &single_entity));

  // Store created eid to make sure the correct one is being returned
  gxf_uid_t old_entity_uid = single_entity;
  single_entity = 0;
  entities_count = 1;
  GXF_ASSERT_SUCCESS(GxfEntityFindAll(context, &entities_count, &single_entity));
  // Should return a single entity with known eid
  GXF_ASSERT_EQ(entities_count, 1);
  GXF_ASSERT_EQ(single_entity, old_entity_uid);

  // Load a subgraph containing multiple entities
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, "gxf/test/unit/test_push_to_transmitter.yaml"));

  // Get number of entities limited by array size
  gxf_uid_t* entities = new gxf_uid_t[4];
  entities_count = 4;
  gxf_result_t find_all_result = GxfEntityFindAll(context, &entities_count, entities);
  GXF_ASSERT_EQ(find_all_result, GXF_QUERY_NOT_ENOUGH_CAPACITY);
  // Returned number of entities should be more than the capacity
  GXF_ASSERT_GT(entities_count, 4);

  delete[] entities;

  // Get number of entities not limited by array size
  entities = new gxf_uid_t[32];
  entities_count = 32;
  GXF_ASSERT_SUCCESS(GxfEntityFindAll(context, &entities_count, entities));
  GXF_ASSERT_GT(entities_count, 4);
  GXF_ASSERT_LT(entities_count, 32);

  // Loop through to ensure eids are all nonzero and to check that the memory wasn't corrupted
  for (uint64_t i = 0; i < entities_count; i++) {
    GXF_ASSERT_NE(entities[i], 0);
  }
  delete[] entities;
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, Param) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));

  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));

  gxf_tid_t tid;
  GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::BlockMemoryPool", &tid));
  gxf_uid_t cid;
  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "test", &cid));

  const char* strings_1 = "something";
  const char* strings_2 = "happened!";
  const char* strings[] = {strings_1, strings_2};
  double float_1d_vector[4] = {101.0, 1200.0, 101.0, 121.0};
  int64_t int64_2d_vector_1[3] = {9, 7, 4};
  int64_t int64_2d_vector_2[3] = {100, 102, 101};
  int64_t *int64_2d_vector[2];
  int64_2d_vector[0] = int64_2d_vector_1;
  int64_2d_vector[1] = int64_2d_vector_2;
  GXF_ASSERT_SUCCESS(GxfParameterSetFloat64(context, cid, "float64_var", 11.11));
  GXF_ASSERT_SUCCESS(GxfParameterSetFloat32(context, cid, "float32_var", 2.0));
  GXF_ASSERT_SUCCESS(GxfParameterSetInt64(context, cid, "int64_var", 111));
  GXF_ASSERT_SUCCESS(GxfParameterSetStr(context, cid, "str_var", "string_param"));
  GXF_ASSERT_SUCCESS(GxfParameterSetUInt64(context, cid, "uint64_var", 111));
  GXF_ASSERT_SUCCESS(GxfParameterSetUInt32(context, cid, "uint32_var", 123));
  GXF_ASSERT_SUCCESS(GxfParameterSetUInt16(context, cid, "uint16_var", 235));
  GXF_ASSERT_SUCCESS(GxfParameterSetBool(context, cid, "bool_var", true));
  GXF_ASSERT_SUCCESS(GxfParameterSetInt32(context, cid, "int32_var", 2147483647));
  GXF_ASSERT_SUCCESS(GxfParameterSet1DStrVector(context, cid, "str_1d_vector", strings, 2));
  GXF_ASSERT_SUCCESS(GxfParameterSet1DFloat64Vector(context, cid, "float64_vector", float_1d_vector, 4));
  GXF_ASSERT_SUCCESS(GxfParameterSet2DInt64Vector(context, cid, "int64_2d_vector", int64_2d_vector, 2, 3));
  double float64_var;
  float float32_var;
  int64_t int64_value;
  uint64_t uint64_value;
  uint32_t uint32_value;
  uint16_t uint16_value;
  bool bool_value = false;
  int32_t int32_value;
  const char* str_value;
  char* strings_value_1 = new char[10]();
  char* strings_value_2 = new char[10]();
  char* strings_value[2] = {strings_value_1, strings_value_2};
  uint64_t count = 2;
  uint64_t min_length = 10;
  uint64_t float_1d_vector_length = 6;
  uint64_t int64_2d_vector_height = 2;
  uint64_t int64_2d_vector_width = 6;
  double float_1d_vector_value[float_1d_vector_length];
  int64_t int64_2d_vector_value_1[int64_2d_vector_width];
  int64_t int64_2d_vector_value_2[int64_2d_vector_width];
  int64_t *int64_2d_vector_value[2];
  int64_2d_vector_value[0] = int64_2d_vector_value_1;
  int64_2d_vector_value[1] = int64_2d_vector_value_2;
  GXF_ASSERT_SUCCESS(GxfParameterGetFloat64(context, cid, "float64_var", &float64_var));
  GXF_ASSERT_SUCCESS(GxfParameterGetFloat32(context, cid, "float32_var", &float32_var));
  GXF_ASSERT_SUCCESS(GxfParameterGetInt64(context, cid, "int64_var", &int64_value));
  GXF_ASSERT_SUCCESS(GxfParameterGetUInt64(context, cid, "uint64_var", &uint64_value));
  GXF_ASSERT_SUCCESS(GxfParameterGetUInt64(context, cid, "uint64_var", &uint64_value));
  GXF_ASSERT_SUCCESS(GxfParameterGetUInt32(context, cid, "uint32_var", &uint32_value));
  GXF_ASSERT_SUCCESS(GxfParameterGetUInt16(context, cid, "uint16_var", &uint16_value));
  GXF_ASSERT_SUCCESS(GxfParameterGetStr(context, cid, "str_var", &str_value));
  GXF_ASSERT_SUCCESS(GxfParameterGetBool(context, cid, "bool_var", &bool_value));
  GXF_ASSERT_SUCCESS(GxfParameterGetInt32(context, cid, "int32_var", &int32_value));
  GXF_ASSERT_SUCCESS(GxfParameterGet1DStrVector(context, cid, "str_1d_vector", strings_value, &count, &min_length));
  GXF_ASSERT_SUCCESS(GxfParameterGet1DFloat64Vector(
      context, cid, "float64_vector", float_1d_vector_value, &float_1d_vector_length));
  GXF_ASSERT_SUCCESS(GxfParameterGet2DInt64Vector(context, cid, "int64_2d_vector",
                                                  int64_2d_vector_value, &int64_2d_vector_height,
                                                  &int64_2d_vector_width));
  GXF_ASSERT_EQ(float64_var, 11.11);
  GXF_ASSERT_EQ(float32_var, 2.0);
  GXF_ASSERT_EQ(int64_value, 111);
  GXF_ASSERT_EQ(uint64_value, 111);
  GXF_ASSERT_EQ(uint32_value, 123);
  GXF_ASSERT_EQ(uint16_value, 235);
  GXF_ASSERT_STREQ(str_value, "string_param");
  GXF_ASSERT_EQ(bool_value, true);
  GXF_ASSERT_EQ(int32_value, 2147483647);
  GXF_ASSERT_STREQ(strings_value[0], "something");
  GXF_ASSERT_STREQ(strings_value[1], "happened!");
  GXF_ASSERT_EQ(count, 2);
  GXF_ASSERT_EQ(min_length, 9);
  GXF_ASSERT_EQ(float_1d_vector_length, 4);
  GXF_ASSERT_EQ(int64_2d_vector_height, 2);
  GXF_ASSERT_EQ(int64_2d_vector_width, 3);
  GXF_ASSERT_EQ(float_1d_vector_value[0], 101.00);
  GXF_ASSERT_EQ(float_1d_vector_value[1], 1200.0);
  GXF_ASSERT_EQ(float_1d_vector_value[2], 101.0);
  GXF_ASSERT_EQ(float_1d_vector_value[3], 121.0);
  GXF_ASSERT_EQ(int64_2d_vector_value[0][0], 9);
  GXF_ASSERT_EQ(int64_2d_vector_value[0][1], 7);
  GXF_ASSERT_EQ(int64_2d_vector_value[0][2], 4);
  GXF_ASSERT_EQ(int64_2d_vector_value[1][0], 100);
  GXF_ASSERT_EQ(int64_2d_vector_value[1][1], 102);
  GXF_ASSERT_EQ(int64_2d_vector_value[1][2], 101);
  GXF_ASSERT_EQ(GxfParameterGet2DInt64Vector(context, cid, "int64_2d_vector", NULL,
                                             &int64_2d_vector_height, &int64_2d_vector_width),
                GXF_ARGUMENT_NULL);
  GXF_ASSERT_EQ(GxfParameterSet2DInt64Vector(context, cid, "int64_2d_vector", NULL, 4, 4),
                GXF_ARGUMENT_NULL);
  GXF_ASSERT_EQ(GxfParameterSet1DInt64Vector(context, cid, "int64_1d_vector", NULL, 4),
                GXF_ARGUMENT_NULL);
  int64_2d_vector_height = 1;
  int64_2d_vector_width = 1;
  GXF_ASSERT_EQ(GxfParameterGet2DInt64Vector(context, cid, "int64_2d_vector", int64_2d_vector_value,
                                             &int64_2d_vector_height, &int64_2d_vector_width),
                GXF_QUERY_NOT_ENOUGH_CAPACITY);
  GXF_ASSERT_EQ(int64_2d_vector_height, 2);
  GXF_ASSERT_EQ(int64_2d_vector_width, 3);
  GXF_ASSERT_SUCCESS(GxfParameterGet2DInt64Vector(context, cid, "int64_2d_vector",
                                                  int64_2d_vector_value, &int64_2d_vector_height,
                                                  &int64_2d_vector_width));
  GXF_ASSERT_SUCCESS(GxfParameterSet2DInt64Vector(context, cid, "int64_2d_vector", NULL, 4, 0));
  GXF_ASSERT_SUCCESS(GxfParameterGet2DInt64Vector(context, cid, "int64_2d_vector",
                                                  int64_2d_vector_value, &int64_2d_vector_height,
                                                  &int64_2d_vector_width));
  GXF_ASSERT_EQ(int64_2d_vector_height, 4);
  GXF_ASSERT_EQ(int64_2d_vector_width, 0);

  // Test unavailable parameter
  GXF_ASSERT_EQ(GxfParameterGetFloat64(context, cid, "float64_var1", &float64_var),
                GXF_PARAMETER_NOT_FOUND);

  GXF_ASSERT_SUCCESS(GxfEntityDestroy(context, eid));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));

  delete [] strings_value_1;
  delete [] strings_value_2;
}

TEST(Entity, PushToReceiver) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));

  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, "gxf/test/unit/test_push_to_transmitter.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));

  gxf_uid_t eid;
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "rx", &eid));

  gxf_tid_t tid;
  GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::DoubleBufferReceiver", &tid));

  int offset = 0;
  gxf_uid_t cid;
  GXF_ASSERT_SUCCESS(GxfComponentFind(context, eid, tid, "signal", &offset, &cid));
  auto rx = nvidia::gxf::Handle<nvidia::gxf::DoubleBufferReceiver>::Create(context, cid);
  GXF_ASSERT_TRUE(rx.has_value());

  for (int i = 0; i < 10; i++) {
    auto message = nvidia::gxf::Entity::New(context);
    rx.value()->push(std::move(message.value()));
    rx.value()->sync();
  }

  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, EntityStatus) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));

  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, "gxf/test/unit/test_push_to_transmitter.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));

  gxf_uid_t eid;
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "rx", &eid));
  gxf_entity_status_t entity_status;
  GXF_ASSERT_SUCCESS(GxfEntityGetStatus(context, eid, &entity_status));
  GXF_ASSERT_EQ(entity_status, GXF_ENTITY_STATUS_NOT_STARTED);

  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));

  gxf_tid_t tid;
  GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::DoubleBufferReceiver", &tid));
  int offset = 0;
  gxf_uid_t cid;
  GXF_ASSERT_SUCCESS(GxfComponentFind(context, eid, tid, "signal", &offset, &cid));
  auto rx = nvidia::gxf::Handle<nvidia::gxf::DoubleBufferReceiver>::Create(context, cid);
  GXF_ASSERT_TRUE(rx.has_value());

  for (int i = 0; i < 10; i++) {
    auto message = nvidia::gxf::Entity::New(context);
    rx.value()->push(std::move(message.value()));
    rx.value()->sync();
  }

  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

// Loads a graph file which has unspecified handle components. Ideally these
// unspecified handles must be set in a subsequent parameter file like
// "gxf/test/apps/test_ping_graph_parameters.yaml"
TEST(Entity, UnspecifiedHandle) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));

  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, "gxf/test/apps/test_ping_graph.yaml"));
  GXF_ASSERT_EQ(GxfGraphActivate(context), GXF_PARAMETER_MANDATORY_NOT_SET);
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, GxfGraphParamOverride) {
  gxf_context_t context;
  const char* param_override_string[2];
  char param1[32] = "tx/ping_tx/signal=tx/signal";
  char param2[32] = "rx/ping_rx/signal=rx/signal";
  param_override_string[0] = &param1[0];
  param_override_string[1] = &param2[0];
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));

  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, "gxf/test/apps/test_ping_graph.yaml",
                                   &param_override_string[0], 2));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));

  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, MultipleEntites) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));

  const GxfLoadExtensionsInfo info{&kStdExtensionFilename, 1, nullptr, 0, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));

  gxf_uid_t eid1 = kNullUid;
  const GxfEntityCreateInfo entity_create_info = {"SameNameEntity"};
  GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid1));

  gxf_uid_t eid2 = kNullUid;
  GXF_ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid2), GXF_ARGUMENT_INVALID);

  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, GxfGraphStartOrder) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));

  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, "gxf/test/apps/test_ping_multi_thread_start_order.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}
