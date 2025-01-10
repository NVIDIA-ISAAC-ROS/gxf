/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <chrono>
#include <climits>
#include <cstring>
#include <iostream>
#include <thread>

#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/app/entity_group.hpp"
#include "gxf/app/arg.hpp"
#include "gxf/app/graph_entity.hpp"
#include "gxf/core/gxf.h"
#include "gxf/sample/multi_ping_rx.hpp"
#include "gxf/sample/ping_rx.hpp"
#include "gxf/sample/ping_tx.hpp"
#include "gxf/std/clock.hpp"
#include "gxf/std/double_buffer_receiver.hpp"
#include "gxf/std/double_buffer_transmitter.hpp"
#include "gxf/std/scheduling_terms.hpp"

namespace {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

}  // namespace

namespace nvidia {
namespace gxf {

class GxfEntityGroup : public ::testing::Test {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GxfSetSeverity(context, GXF_SEVERITY_DEBUG);
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  }

  void TearDown() { GXF_ASSERT_SUCCESS(GxfContextDestroy(context)); }

 protected:
  gxf_context_t context = kNullContext;
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  const GxfEntityCreateInfo entity_create_info = {0};
  struct TestEntity {
    GraphEntityPtr entity = std::make_shared<GraphEntity>();
    const char* name_entity_group = "UNKNOWN";
    uint64_t resource_num = 0;
  };
};

TEST_F(GxfEntityGroup, Basic) {
  TestEntity e1, e2, e3;
  e1.entity->setup(context, "basic1");
  e2.entity->setup(context, "basic2");
  e3.entity->setup(context, "basic3");

  GxfEntityGroupName(context, e1.entity->eid(), &e1.name_entity_group);
  GxfEntityGroupName(context, e2.entity->eid(), &e2.name_entity_group);
  GxfEntityGroupName(context, e3.entity->eid(), &e3.name_entity_group);

  GXF_ASSERT_STREQ(e1.name_entity_group, kDefaultEntityGroupName);
  GXF_ASSERT_STREQ(e2.name_entity_group, kDefaultEntityGroupName);
  GXF_ASSERT_STREQ(e3.name_entity_group, kDefaultEntityGroupName);
}

TEST_F(GxfEntityGroup, BasicUserEntityGroup) {
  TestEntity e1, e2, e3;
  e1.entity->setup(context, "basic1");
  e2.entity->setup(context, "basic2");
  e3.entity->setup(context, "basic3");

  GxfEntityGroupName(context, e1.entity->eid(), &e1.name_entity_group);
  GxfEntityGroupName(context, e2.entity->eid(), &e2.name_entity_group);
  GxfEntityGroupName(context, e3.entity->eid(), &e3.name_entity_group);

  GXF_ASSERT_STREQ(e1.name_entity_group, kDefaultEntityGroupName);
  GXF_ASSERT_STREQ(e2.name_entity_group, kDefaultEntityGroupName);
  GXF_ASSERT_STREQ(e3.name_entity_group, kDefaultEntityGroupName);

  const std::string test_entity_group_name = "test_entity_group_1";
  EntityGroup entity_group = EntityGroup();
  entity_group.setup(context, test_entity_group_name.c_str());
  entity_group.add(e1.entity);
  entity_group.add(e2.entity);

  GxfEntityGroupName(context, e1.entity->eid(), &e1.name_entity_group);
  GxfEntityGroupName(context, e2.entity->eid(), &e2.name_entity_group);
  GxfEntityGroupName(context, e3.entity->eid(), &e3.name_entity_group);

  GXF_ASSERT_STREQ(e1.name_entity_group, test_entity_group_name.c_str());
  GXF_ASSERT_STREQ(e2.name_entity_group, test_entity_group_name.c_str());
  GXF_ASSERT_STREQ(e3.name_entity_group, kDefaultEntityGroupName);
}

TEST_F(GxfEntityGroup, BasicUserEntityGroupAddAsList) {
  TestEntity e1, e2, e3;
  e1.entity->setup(context, "basic1");
  e2.entity->setup(context, "basic2");
  e3.entity->setup(context, "basic3");

  GxfEntityGroupName(context, e1.entity->eid(), &e1.name_entity_group);
  GxfEntityGroupName(context, e2.entity->eid(), &e2.name_entity_group);
  GxfEntityGroupName(context, e3.entity->eid(), &e3.name_entity_group);

  GXF_ASSERT_STREQ(e1.name_entity_group, kDefaultEntityGroupName);
  GXF_ASSERT_STREQ(e2.name_entity_group, kDefaultEntityGroupName);
  GXF_ASSERT_STREQ(e3.name_entity_group, kDefaultEntityGroupName);

  const std::string test_entity_group_name = "test_entity_group_1";
  EntityGroup entity_group = EntityGroup();
  entity_group.setup(context, test_entity_group_name.c_str());
  std::vector<GraphEntityPtr> entity_members = {e1.entity, e2.entity};
  entity_group.add(entity_members);

  GxfEntityGroupName(context, e1.entity->eid(), &e1.name_entity_group);
  GxfEntityGroupName(context, e2.entity->eid(), &e2.name_entity_group);
  GxfEntityGroupName(context, e3.entity->eid(), &e3.name_entity_group);

  GXF_ASSERT_STREQ(e1.name_entity_group, test_entity_group_name.c_str());
  GXF_ASSERT_STREQ(e2.name_entity_group, test_entity_group_name.c_str());
  GXF_ASSERT_STREQ(e3.name_entity_group, kDefaultEntityGroupName);
}

}  // namespace gxf
}  // namespace nvidia
