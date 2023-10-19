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

TEST(GxfEntityFind,null_context)
{
  gxf_context_t context = kNullContext;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  const char* kGraphFileName = "gxf/core/tests/test_ping_dynamic_activation.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context,kGraphFileName));
  std::string Entity_Name="tx0";
  gxf_uid_t eid_rx = kNullUid ;
  ASSERT_EQ(GxfEntityFind(NULL,Entity_Name.c_str(), &eid_rx),GXF_CONTEXT_INVALID);
  ASSERT_EQ(eid_rx,kNullUid);
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityFind,wrong_context)
{
  gxf_context_t context = kNullContext;
  gxf_context_t context1;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context1));
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  const char* kGraphFileName = "gxf/core/tests/test_ping_dynamic_activation.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context,kGraphFileName));
  std::string Entity_Name="tx0";
  gxf_uid_t eid_rx = kNullUid;
  ASSERT_EQ(GxfEntityFind(context1,Entity_Name.c_str(), &eid_rx),GXF_ENTITY_NOT_FOUND);
  ASSERT_EQ(eid_rx,kNullUid);
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityFind,Valid)
{
  gxf_context_t context = kNullContext;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  const char* kGraphFileName = "gxf/core/tests/test_ping_dynamic_activation.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context,kGraphFileName));
  std::string Entity_Name="tx0";
  gxf_uid_t eid_rx = kNullUid;
  GXF_ASSERT_SUCCESS(GxfEntityFind(context,Entity_Name.c_str(), &eid_rx));
  ASSERT_NE(eid_rx,kNullUid);
  GXF_LOG_WARNING("Entity eid for %s eid is %ld", Entity_Name.c_str(), eid_rx);
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityFind,NULL_name)
{
  gxf_context_t context = kNullContext;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  const char* kGraphFileName = "gxf/core/tests/test_ping_dynamic_activation.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context,kGraphFileName));
  std::string  Entity_Name="";
  gxf_uid_t eid_rx = kNullUid;
  GXF_ASSERT_EQ(GxfEntityFind(context,Entity_Name.c_str(), &eid_rx),GXF_ENTITY_NOT_FOUND);
  ASSERT_EQ(eid_rx,kNullUid);
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityFind,invalid_name)
{
  gxf_context_t context = kNullContext;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  const char* kGraphFileName = "gxf/core/tests/test_ping_dynamic_activation.yaml";
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context,kGraphFileName));
  std::string Entity_Name="Invalid_Entity_Name";
  gxf_uid_t eid_tx = kNullUid ;
  GXF_ASSERT_EQ(GxfEntityFind(context,Entity_Name.c_str(),&eid_tx),GXF_ENTITY_NOT_FOUND);
  ASSERT_EQ(eid_tx,kNullUid);
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}
