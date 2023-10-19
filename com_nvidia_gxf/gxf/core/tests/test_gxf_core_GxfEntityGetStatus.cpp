/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

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
constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";
}  // namespace


void entity_status_not_started(gxf_context_t context, std::vector<std::string> entities)
{
    for(auto entity: entities)
    {
      gxf_uid_t eid = kNullUid;
      gxf_entity_status_t entity_status;
      GXF_ASSERT_SUCCESS(GxfEntityFind(context, entity.c_str(), &eid));
      GXF_LOG_WARNING("Entity %s eid is %ld", entity.c_str(), eid);
      GXF_ASSERT_SUCCESS(GxfEntityGetStatus(context,eid,&entity_status));
      ASSERT_EQ(GXF_ENTITY_STATUS_NOT_STARTED,entity_status);
    }
}

TEST(GxfEntityGetStatus,entity_status_not_started)
{
    gxf_context_t context = kNullContext;
    std::vector<std::string> unit_graph {"tx0", "rx0"};
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
    GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, "gxf/core/tests/test_ping_dynamic_activation.yaml"));
    GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    entity_status_not_started(context,unit_graph);
    GXF_ASSERT_SUCCESS(GxfGraphWait(context));
    GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}