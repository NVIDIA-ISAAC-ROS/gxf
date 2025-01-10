/*
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

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

void deactivateEntities_with_null_context(gxf_context_t context, std::vector<std::string> entities)
{
  for(auto entity: entities)
  {
    gxf_uid_t eid = kNullUid;
    GXF_ASSERT_SUCCESS(GxfEntityFind(context, entity.c_str(), &eid));
    ASSERT_NE(eid,kNullUid);
    GXF_LOG_WARNING("Deactivating %s eid is %ld", entity.c_str(), eid);
    ASSERT_EQ(GxfEntityDeactivate(NULL, eid),GXF_CONTEXT_INVALID);
  }
}

void deactivateEntities_with_null_uid(gxf_context_t context, std::vector<std::string> entities)
{
  for(auto entity: entities)
  {
    gxf_uid_t eid = kNullUid;
    GXF_ASSERT_SUCCESS(GxfEntityFind(context, entity.c_str(), &eid));
    ASSERT_NE(eid,kNullUid);
    GXF_LOG_WARNING("Deactivating %s eid is %ld", entity.c_str(), eid);
    ASSERT_EQ(GxfEntityDeactivate(context,kNullUid),GXF_ENTITY_NOT_FOUND);
  }
}

}  // namespace

TEST(GxfEntityDeactivate,deactiving_the_non_activated_entities)
{
   gxf_context_t context = kNullContext;
   GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
   std::vector<std::string> unit_graph {"tx0", "rx0"};
   const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
   GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
   const char* kGraphFileName = "gxf/core/tests/test_ping_dynamic_activation.yaml";
   GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context,kGraphFileName));
   GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
   deactivateEntities(context, unit_graph);
   GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
   GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityDeactivate,deactiving_the_already_activated_entities)
{
   gxf_context_t context = kNullContext;
   GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
   std::vector<std::string> unit_graph {"tx0", "rx0"};
   const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
   GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
   const char* kGraphFileName = "gxf/core/tests/test_ping_dynamic_activation.yaml";
   GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context,kGraphFileName));
   GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
   deactivateEntities(context, unit_graph);
   std::this_thread::sleep_for(std::chrono::milliseconds(50));
   activateEntities(context, unit_graph);
   std::this_thread::sleep_for(std::chrono::milliseconds(50));
   deactivateEntities(context, unit_graph);
   GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
   GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityDeactivate,deactiving_entities_with_null_context)
{
   gxf_context_t context = kNullContext;
   GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
   std::vector<std::string> unit_graph {"tx0", "rx0"};
   const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
   GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
   const char* kGraphFileName = "gxf/core/tests/test_ping_dynamic_activation.yaml";
   GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context,kGraphFileName));
   GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
   deactivateEntities_with_null_context(context, unit_graph);
   GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
   GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityDeactivate,deactiving_entities_with_null_uid)
{
   gxf_context_t context = kNullContext;
   GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
   std::vector<std::string> unit_graph {"tx0", "rx0"};
   const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
   GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
   const char* kGraphFileName = "gxf/core/tests/test_ping_dynamic_activation.yaml";
   GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context,kGraphFileName));
   GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
   deactivateEntities_with_null_uid(context, unit_graph);
   GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
   GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityDeactivate,adding_components_after_deactivating_entities)
{
   gxf_context_t context = kNullContext;
   gxf_tid_t rpi_tid{0xe9234c1ad5f8445c, 0xae9118bcda197032};  // RegisterParameterInterfaceTest
   gxf_uid_t rpi_uid = kNullUid;
   std::string entity_name="rx0";
   gxf_uid_t entity_eid = kNullUid;
   GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
   std::vector<std::string> unit_graph {"tx0", "rx0"};
   const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
   GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
   const char* kGraphFileName = "gxf/core/tests/test_ping_dynamic_activation.yaml";
   GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context,kGraphFileName));
   GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
   deactivateEntities(context, unit_graph);
   GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
   GXF_ASSERT_SUCCESS(GxfEntityFind(context,entity_name.c_str(), &entity_eid));
   ASSERT_NE(kNullUid,entity_eid);
   GXF_ASSERT_SUCCESS(GxfComponentAdd(context,entity_eid, rpi_tid, "rpi", &rpi_uid));
   GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityDeactivate,Mandatory_parameters_changed_after_deactivating_entities)
{
   gxf_context_t context = kNullContext;
   gxf_tid_t rpi_tid{0xe9234c1ad5f8445c, 0xae9118bcda197032};  // RegisterParameterInterfaceTest
   gxf_uid_t rpi_uid = kNullUid;
   std::string entity_name="rx0";
   gxf_uid_t entity_eid = kNullUid;
   GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
   std::vector<std::string> unit_graph {"tx0", "rx0"};
   const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
   GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
   const char* kGraphFileName = "gxf/core/tests/test_ping_dynamic_activation.yaml";
   GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context,kGraphFileName));
   GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
   deactivateEntities(context, unit_graph);
   GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
   GXF_ASSERT_SUCCESS(GxfEntityFind(context,entity_name.c_str(), &entity_eid));
   ASSERT_NE(kNullUid,entity_eid);
   GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::BlockMemoryPool", &rpi_tid));
   GXF_ASSERT_SUCCESS(GxfComponentAdd(context,entity_eid,rpi_tid,"rpi",&rpi_uid));
   GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}