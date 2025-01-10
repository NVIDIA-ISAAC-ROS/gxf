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

namespace {

constexpr const char* kStdExtensionFilename = "gxf/std/libgxf_std.so";
constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

}  // namespace


void activateEntities(gxf_context_t context, std::vector<std::string> entities)
{
  for(auto entity: entities)
  {
    gxf_uid_t eid = kNullUid;
    GXF_ASSERT_SUCCESS(GxfEntityFind(context, entity.c_str(), &eid));
    GXF_LOG_WARNING("Activating %s eid is %ld", entity.c_str(), eid);
    GXF_ASSERT_SUCCESS(GxfEntityActivate(context, eid));
  }
}

void activateEntities_with_null_uid(gxf_context_t context, std::vector<std::string> entities)
{
  for(auto entity: entities)
  {
    gxf_uid_t eid = kNullUid;
    GXF_ASSERT_SUCCESS(GxfEntityFind(context, entity.c_str(), &eid));
    ASSERT_NE(eid,kNullUid);
    GXF_LOG_WARNING("Activating %s eid is %ld", entity.c_str(), eid);
    ASSERT_EQ(GxfEntityActivate(context,kNullUid),GXF_ENTITY_NOT_FOUND);
  }
}

void reactivateEntities(gxf_context_t context, std::vector<std::string> entities)
{
  for(auto entity: entities)
  {
    gxf_uid_t eid = kNullUid;
    GXF_ASSERT_SUCCESS(GxfEntityFind(context, entity.c_str(), &eid));
    ASSERT_NE(eid,kNullUid);
    GXF_LOG_WARNING("Re Activating %s eid is %ld", entity.c_str(), eid);
    ASSERT_EQ(GxfEntityActivate(context, eid),GXF_INVALID_LIFECYCLE_STAGE);
  }
}

void deactivateEntities(gxf_context_t context, std::vector<std::string> entities)
{
  for(auto entity: entities)
  {
    gxf_uid_t eid = kNullUid;
    GXF_ASSERT_SUCCESS(GxfEntityFind(context, entity.c_str(), &eid));
    GXF_LOG_WARNING("Deactivating %s eid is %ld", entity.c_str(), eid);
    GXF_ASSERT_SUCCESS(GxfEntityDeactivate(context, eid));
  }
}

TEST(GxfEntityActivate,Mandatory_parameters_not_set) {
  gxf_context_t context = kNullContext;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  constexpr const char* kExtensions[] = {
    "gxf/std/libgxf_std.so",
    "gxf/test/extensions/libgxf_test.so",
  };
  const GxfLoadExtensionsInfo info{kExtensions, 2, nullptr, 0, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  gxf_uid_t eid = kNullUid;
  const GxfEntityCreateInfo entity_create_info = {0};
  GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));
  gxf_tid_t rpi_tid{0xe9234c1ad5f8445c, 0xae9118bcda197032};  // RegisterParameterInterfaceTest
  gxf_uid_t rpi_uid = kNullUid;
  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, rpi_tid, "rpi", &rpi_uid));
  GXF_ASSERT_EQ(GxfEntityActivate(context, eid), GXF_PARAMETER_MANDATORY_NOT_SET);
  GXF_ASSERT_EQ(GxfComponentRemove(context, eid, rpi_tid, "rpi"), GXF_ENTITY_NOT_FOUND);
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityActivate,Mandatory_parameters_changed_after_activating_entities)
{
   gxf_context_t context = kNullContext;
   gxf_tid_t rpi_tid{0xe9234c1ad5f8445c, 0xae9118bcda197032};  // RegisterParameterInterfaceTest
   gxf_uid_t rpi_uid = kNullUid;
   std::string entity_name="rx0";
   std::vector<std::string> unit_graph {"tx0", "rx0"};
   gxf_uid_t entity_eid = kNullUid;
   GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
   const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
   GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
   GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, "gxf/core/tests/test_ping_dynamic_activation.yaml"));
   GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
   GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
   std::this_thread::sleep_for(std::chrono::milliseconds(5));
   deactivateEntities(context, unit_graph);
   std::this_thread::sleep_for(std::chrono::milliseconds(5));
   activateEntities(context, unit_graph);
   GXF_ASSERT_SUCCESS(GxfEntityFind(context,entity_name.c_str(), &entity_eid));
   ASSERT_NE(kNullUid,entity_eid);
   GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::BlockMemoryPool", &rpi_tid));
   GXF_ASSERT_EQ(GxfComponentAdd(context,entity_eid,rpi_tid,"rpi",&rpi_uid),GXF_ENTITY_CAN_NOT_ADD_COMPONENT_AFTER_INITIALIZATION);
   GXF_ASSERT_EQ(GxfComponentRemove(context,entity_eid,rpi_tid,"rpi"),GXF_ENTITY_COMPONENT_NOT_FOUND);
   GXF_ASSERT_SUCCESS(GxfGraphWait(context));
   GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
   GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}
