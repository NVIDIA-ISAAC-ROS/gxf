/*
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"
#include <string>

namespace {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

}  // namespace

class GxfUpdateEntityGroup_Test : public ::testing::Test {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context_));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context_, &info_));
    // 1. create empty entities representing workload entities
    for (int i = 0; i < num_work_entities_; ++i) {
      EntityInfo entity_info;
      entity_info.name = "workload" + std::to_string(i);
      entity_info.entity_create_info = {entity_info.name.c_str(), GXF_ENTITY_CREATE_PROGRAM_BIT};
      GXF_ASSERT_SUCCESS(GxfCreateEntity(context_, &entity_info.entity_create_info, &entity_info.eid));
      GXF_ASSERT_NE(entity_info.eid, kNullUid);
      workload_entities_.push_back(entity_info);
    }
    // 2. create entities with resource prefilled representing resource entities
    gxf_tid_t tid = GxfTidNull();
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context_, "nvidia::gxf::ResourceBase", &tid));
    for (int i = 0; i < num_resource_entities_; ++i) {
      EntityInfo entity_info;
      entity_info.name = "resource_e" + std::to_string(i);
      entity_info.entity_create_info = {entity_info.name.c_str(), GXF_ENTITY_CREATE_PROGRAM_BIT};
      GXF_ASSERT_SUCCESS(GxfCreateEntity(context_, &entity_info.entity_create_info, &entity_info.eid));
      GXF_ASSERT_NE(entity_info.eid, kNullUid);

      gxf_uid_t cid = kNullUid;
      std::string comp_name = "resource_c" + std::to_string(i);
      GXF_ASSERT_SUCCESS(GxfComponentAdd(context_, entity_info.eid, tid, comp_name.c_str(), &cid));
      GXF_ASSERT_NE(cid, kNullUid);
      resource_entities_.push_back(entity_info);
    }
    // 3. create empty entity groups
    for (int i = 0; i < num_groups_; ++i) {
      EntityGroupInfo group_info = {
        .name = "group" + std::to_string(i)
      };
      GXF_ASSERT_SUCCESS(GxfCreateEntityGroup(context_, group_info.name.c_str(), &group_info.gid));
      GXF_ASSERT_NE(group_info.gid, kNullUid);
      entity_groups_.push_back(group_info);
    }
  }

  void TearDown() {
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context_));
  }

 protected:
  gxf_context_t context_ = kNullContext;
  const GxfLoadExtensionsInfo info_{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  struct EntityInfo {
    GxfEntityCreateInfo entity_create_info;
    gxf_uid_t eid = kNullUid;
    std::string name;
  };
  struct EntityGroupInfo {
    std::string name;
    gxf_uid_t gid = kNullUid;
  };
  int num_work_entities_ = 6;
  int num_resource_entities_ = 4;
  int num_groups_ = 2;
  std::vector<EntityInfo> workload_entities_;
  std::vector<EntityInfo> resource_entities_;
  std::vector<EntityGroupInfo> entity_groups_;
};

TEST_F(GxfUpdateEntityGroup_Test, ValidWorkloadEntities) {
  for (int i = 0; i < num_work_entities_; ++i) {
    if (i % 2 == 0) {
      GXF_ASSERT_SUCCESS(GxfUpdateEntityGroup(context_, entity_groups_.at(0).gid, workload_entities_.at(i).eid));
    } else if (i % 2 == 1) {
      GXF_ASSERT_SUCCESS(GxfUpdateEntityGroup(context_, entity_groups_.at(1).gid, workload_entities_.at(i).eid));
    }
  }
}

TEST_F(GxfUpdateEntityGroup_Test, ValidResourceEntities) {
  for (int i = 0; i < num_resource_entities_; ++i) {
    if (i % 2 == 0) {
      GXF_ASSERT_SUCCESS(GxfUpdateEntityGroup(context_, entity_groups_.at(0).gid, resource_entities_.at(i).eid));
    } else if (i % 2 == 1) {
      GXF_ASSERT_SUCCESS(GxfUpdateEntityGroup(context_, entity_groups_.at(1).gid, resource_entities_.at(i).eid));
    }
  }
}

TEST_F(GxfUpdateEntityGroup_Test, ValidBothWorkloadAndResourceEntities) {
  for (int i = 0; i < num_work_entities_; ++i) {
    if (i % 2 == 0) {
      GXF_ASSERT_SUCCESS(GxfUpdateEntityGroup(context_, entity_groups_.at(0).gid, workload_entities_.at(i).eid));
    } else if (i % 2 == 1) {
      GXF_ASSERT_SUCCESS(GxfUpdateEntityGroup(context_, entity_groups_.at(1).gid, workload_entities_.at(i).eid));
    }
  }
  for (int i = 0; i < num_resource_entities_; ++i) {
    if (i % 2 == 0) {
      GXF_ASSERT_SUCCESS(GxfUpdateEntityGroup(context_, entity_groups_.at(0).gid, resource_entities_.at(i).eid));
    } else if (i % 2 == 1) {
      GXF_ASSERT_SUCCESS(GxfUpdateEntityGroup(context_, entity_groups_.at(1).gid, resource_entities_.at(i).eid));
    }
  }
}

TEST_F(GxfUpdateEntityGroup_Test, InvalidEntityGroup) {
  gxf_uid_t gid_invalid = ++num_groups_;
  GXF_ASSERT_EQ(GxfUpdateEntityGroup(context_, gid_invalid, workload_entities_.at(0).eid), GXF_ENTITY_GROUP_NOT_FOUND);
}

TEST_F(GxfUpdateEntityGroup_Test, InvalidEntity) {
  gxf_uid_t eid_invalid = 1 + 2 * num_resource_entities_ + num_work_entities_ + num_groups_;
  GXF_ASSERT_EQ(GxfUpdateEntityGroup(context_, entity_groups_.at(0).gid, eid_invalid), GXF_ENTITY_NOT_FOUND);
}