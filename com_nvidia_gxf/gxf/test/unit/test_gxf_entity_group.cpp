/*
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <chrono>
#include <cstring>
#include <thread>
#include <map>

#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"
#include "gxf/core/handle.hpp"
#include "gxf/sample/ping_tx.hpp"

namespace {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

}  // namespace

class EntityGroupTestBase : public ::testing::Test {
 protected:
  std::string test_graph_file_;
  // Modify this table for different test graphs
  virtual bool initEntitiesExpected() {
    test_graph_file_ = "gxf/test/apps/test_entity_group_root_and_subgraph.yaml";
    entities_expected_.emplace("test_subgraph.codelet0", TestEntity{kNullUid, "subgraph_EG_0", 1});
    entities_expected_.emplace("test_subgraph.codelet1", TestEntity{kNullUid, "subgraph_EG_1", 1});
    entities_expected_.emplace("test_subgraph.codelet2", TestEntity{kNullUid, "rootgraph_EG_1", 2});
    entities_expected_.emplace("message_generator", TestEntity{kNullUid, "rootgraph_EG_0", 2});
    entities_expected_.emplace("message_sink", TestEntity{kNullUid, "rootgraph_EG_1", 2});
    return true;
  }
 public:
  void SetUp() {
    GXF_ASSERT_EQ(initEntitiesExpected(), true);
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context_));
    // GXF_ASSERT_SUCCESS(GxfSetSeverity(context_, static_cast<gxf_severity_t>(4)));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context_, &info_));
    GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context_, test_graph_file_.c_str()));
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context_, "nvidia::gxf::ThreadPool", &tid_thread_pool_));
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context_, "nvidia::gxf::GPUDevice", &tid_gpu_device_));
    for (auto& it : entities_expected_) {
      entities_actual_.emplace(it.first, TestEntity{kNullUid, ""});
      auto it_actual = entities_actual_.find(it.first);
      GXF_ASSERT_SUCCESS(GxfEntityFind(context_, it_actual->first.c_str(), &it_actual->second.eid));
      it.second.eid = it_actual->second.eid;
      GXF_ASSERT_SUCCESS(GxfEntityGroupName(context_, it_actual->second.eid, &it_actual->second.name_entity_group));
    }
    GXF_ASSERT_SUCCESS(GxfGraphActivate(context_));
    GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context_));
  }

  void TearDown() {
    GXF_ASSERT_SUCCESS(GxfGraphWait(context_));
    GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context_));
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context_));
  }

 protected:
  gxf_context_t context_ = kNullContext;
  const GxfLoadExtensionsInfo info_{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  const char* kGPUDeviceType = nvidia::TypenameAsString<nvidia::gxf::GPUDevice>();
  const char* kThreadPoolType = nvidia::TypenameAsString<nvidia::gxf::ThreadPool>();
  gxf_tid_t tid_thread_pool_;
  gxf_tid_t tid_gpu_device_;
  struct TestEntity {
    gxf_uid_t eid = kNullUid;
    const char* name_entity_group = "UNKNOWN";
    uint64_t resource_num = 0;
  };
  std::map<std::string, TestEntity> entities_expected_;
  std::map<std::string, TestEntity> entities_actual_;
  bool assertEntityGroupNameEQ(std::string entity_name) {
    GXF_ASSERT_STREQ(entities_actual_[entity_name].name_entity_group, entities_actual_[entity_name].name_entity_group);
    return true;
  }
  bool isResourceTid(gxf_tid_t tid) {
    if (tid == tid_thread_pool_ || tid == tid_gpu_device_) {
      return true;
    } else {
      return false;
    }
  }
  bool assertResourceNumEQ(std::string entity_name) {
    gxf_uid_t eid = entities_expected_[entity_name].eid;
    uint64_t resource_num = entities_expected_[entity_name].resource_num;
    gxf_uid_t resource_cids[kMaxComponents];
    uint64_t num_resource_cids = kMaxComponents;
    GXF_ASSERT_SUCCESS(GxfEntityGroupFindResources(context_, eid, &num_resource_cids, resource_cids));
    GXF_ASSERT_EQ(num_resource_cids, resource_num);
    for (size_t i = 0; i < num_resource_cids; i++) {
      gxf_uid_t resource_cid = resource_cids[i];
      const char *resource_name;
      gxf_tid_t resource_tid;
      GXF_ASSERT_SUCCESS(GxfComponentName(context_, resource_cid, &resource_name));
      GXF_ASSERT_SUCCESS(GxfComponentType(context_, resource_cid, &resource_tid));
      GXF_ASSERT_TRUE(isResourceTid(resource_tid));
    }
    return true;
  }
};

//
// Test Default EntityGroups
//
class EntityGroupDefaultGroup_Test : public EntityGroupTestBase {
 protected:
  // Modify this table if graph changes
  virtual bool initEntitiesExpected() override {
    test_graph_file_ = "gxf/test/apps/test_entity_group_default.yaml";
    entities_expected_.emplace("tx0", TestEntity{kNullUid, kDefaultEntityGroupName, 4});
    entities_expected_.emplace("rx0", TestEntity{kNullUid, kDefaultEntityGroupName, 4});
    entities_expected_.emplace("tx1", TestEntity{kNullUid, kDefaultEntityGroupName, 4});
    entities_expected_.emplace("rx1", TestEntity{kNullUid, kDefaultEntityGroupName, 4});
    return true;
  }
};

TEST_F(EntityGroupDefaultGroup_Test, EntityGroupName) {
  assertEntityGroupNameEQ("tx0");
  assertEntityGroupNameEQ("rx0");
  assertEntityGroupNameEQ("tx1");
  assertEntityGroupNameEQ("rx1");
}

TEST_F(EntityGroupDefaultGroup_Test, EntityGroupResourceCount) {
  assertResourceNumEQ("tx0");
  assertResourceNumEQ("rx0");
  assertResourceNumEQ("tx1");
  assertResourceNumEQ("rx1");
}

TEST_F(EntityGroupDefaultGroup_Test, ResourceMapping) {
  gxf_uid_t cid_0;
  GXF_ASSERT_SUCCESS(GxfEntityResourceGetHandle(context_, entities_actual_["tx0"].eid, kGPUDeviceType, "GPU_0", &cid_0));
  auto maybe_handle_0 = nvidia::gxf::Handle<nvidia::gxf::GPUDevice>::Create(context_, cid_0);
  ASSERT_TRUE(maybe_handle_0) << "GPUDevice resource is not found";
  GXF_ASSERT_EQ(maybe_handle_0.value()->device_id(), 0);

  gxf_uid_t cid_1;
  GXF_ASSERT_SUCCESS(GxfEntityResourceGetHandle(context_, entities_actual_["tx0"].eid, kGPUDeviceType, "GPU_1", &cid_1));
  auto maybe_handle_1 = nvidia::gxf::Handle<nvidia::gxf::GPUDevice>::Create(context_, cid_1);
  ASSERT_TRUE(maybe_handle_1) << "GPUDevice resource is not found";
  GXF_ASSERT_EQ(maybe_handle_1.value()->device_id(), 1);

  gxf_uid_t cid_2;
  GXF_ASSERT_SUCCESS(GxfEntityResourceGetHandle(context_, entities_actual_["tx0"].eid, kThreadPoolType, "ThP_0", &cid_2));
  auto maybe_handle_2 = nvidia::gxf::Handle<nvidia::gxf::ThreadPool>::Create(context_, cid_2);
  ASSERT_TRUE(maybe_handle_2) << "GPUDevice resource is not found";
  GXF_ASSERT_EQ(maybe_handle_2.value()->size(), 5);

  gxf_uid_t cid_3;
  GXF_ASSERT_SUCCESS(GxfEntityResourceGetHandle(context_, entities_actual_["tx0"].eid, kThreadPoolType, "ThP_1", &cid_3));
  auto maybe_handle_3 = nvidia::gxf::Handle<nvidia::gxf::ThreadPool>::Create(context_, cid_3);
  ASSERT_TRUE(maybe_handle_3) << "GPUDevice resource is not found";
  GXF_ASSERT_EQ(maybe_handle_3.value()->size(), 1);
}

//
// Test User defined EntityGroups
//
class EntityGroupUserGroup_Test : public EntityGroupTestBase {
 protected:
  // Modify this table if graph changes
  virtual bool initEntitiesExpected() override {
    test_graph_file_ = "gxf/test/apps/test_entity_group_users.yaml";
    entities_expected_.emplace("tx0", TestEntity{kNullUid, "EG_0", 2});
    entities_expected_.emplace("rx0", TestEntity{kNullUid, "EG_1", 2});
    entities_expected_.emplace("tx1", TestEntity{kNullUid, "EG_0", 2});
    entities_expected_.emplace("rx1", TestEntity{kNullUid, "EG_1", 2});
    return true;
  }
};

TEST_F(EntityGroupUserGroup_Test, EntityGroupName) {
  assertEntityGroupNameEQ("tx0");
  assertEntityGroupNameEQ("rx0");
  assertEntityGroupNameEQ("tx1");
  assertEntityGroupNameEQ("rx1");
}

TEST_F(EntityGroupUserGroup_Test, EntityGroupResourceCount) {
  assertResourceNumEQ("tx0");
  assertResourceNumEQ("rx0");
  assertResourceNumEQ("tx1");
  assertResourceNumEQ("rx1");
}

//
// Test EntityGroup update between Default Group and User Group
//
class EntityGroupDefaultAndUserGroup_Test : public EntityGroupTestBase {
 protected:
  // Modify this table if graph changes
  virtual bool initEntitiesExpected() override {
    test_graph_file_ = "gxf/test/apps/test_entity_group_default_and_users.yaml";
    entities_expected_.emplace("tx0", TestEntity{kNullUid, kDefaultEntityGroupName, 2});
    entities_expected_.emplace("rx0", TestEntity{kNullUid, kDefaultEntityGroupName, 2});
    entities_expected_.emplace("tx1", TestEntity{kNullUid, "EG_0", 2});
    entities_expected_.emplace("rx1", TestEntity{kNullUid, "EG_1", 2});
    return true;
  }
};

TEST_F(EntityGroupDefaultAndUserGroup_Test, EntityGroupName) {
  assertEntityGroupNameEQ("tx0");
  assertEntityGroupNameEQ("rx0");
  assertEntityGroupNameEQ("tx1");
  assertEntityGroupNameEQ("rx1");
}

TEST_F(EntityGroupDefaultAndUserGroup_Test, EntityGroupResourceCount) {
  assertResourceNumEQ("tx0");
  assertResourceNumEQ("rx0");
  assertResourceNumEQ("tx1");
  assertResourceNumEQ("rx1");
}

//
// Test EntityGroup update between parent graph and subgraph
//
class EntityGroupSubgraph_Test : public EntityGroupTestBase {
 protected:
  // Modify this table if graph changes
  virtual bool initEntitiesExpected() {
    test_graph_file_ = "gxf/test/apps/test_entity_group_root_and_subgraph.yaml";
    entities_expected_.emplace("test_subgraph.codelet0", TestEntity{kNullUid, "subgraph_EG_0", 1});
    entities_expected_.emplace("test_subgraph.codelet1", TestEntity{kNullUid, "subgraph_EG_1", 1});
    entities_expected_.emplace("test_subgraph.codelet2", TestEntity{kNullUid, "rootgraph_EG_1", 2});
    entities_expected_.emplace("message_generator", TestEntity{kNullUid, "rootgraph_EG_0", 2});
    entities_expected_.emplace("message_sink", TestEntity{kNullUid, "rootgraph_EG_1", 2});
    return true;
  }
};

TEST_F(EntityGroupSubgraph_Test, EntityGroupName) {
  // Subgraph keeps its EntityGroup
  assertEntityGroupNameEQ("test_subgraph.codelet0");
  assertEntityGroupNameEQ("test_subgraph.codelet1");

  // Parent graph overwrites subgraph EntityGroup
  assertEntityGroupNameEQ("test_subgraph.codelet2");

  // Parent graph keeps its EntityGroup
  assertEntityGroupNameEQ("message_generator");
  assertEntityGroupNameEQ("message_sink");
}

TEST_F(EntityGroupSubgraph_Test, EntityGroupResourceCount) {
  // Subgraph EntityGroup resource count
  assertResourceNumEQ("test_subgraph.codelet0");
  assertResourceNumEQ("test_subgraph.codelet1");

  // Parent graph overwrites subgraph EntityGroup providing new resource count
  assertResourceNumEQ("test_subgraph.codelet2");

  // Parent graph resource count
  assertResourceNumEQ("message_generator");
  assertResourceNumEQ("message_sink");
}