/*
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <thread>

#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"
#include "gxf/core/entity.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/std/default_extension.hpp"
#include "gxf/std/block_memory_pool.hpp"

using nvidia::gxf::ToResultCode;
namespace {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";
constexpr const char* kStdExtensionFilename = "gxf/std/libgxf_std.so";
constexpr const char* kCorePropertyRefCount = "__ref_count";

}  // namespace

class EntityThreadSafe_Test : public ::testing::Test {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context_));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context_, &info_));
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context_, "nvidia::gxf::BlockMemoryPool", &tid));
    // GXF_ASSERT_SUCCESS(GxfSetSeverity(context_, static_cast<gxf_severity_t>(4)));
  }
  void TearDown() {
    // don't forget to clear entities before destroy context
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context_));
  }
 protected:
  gxf_context_t context_ = kNullContext;
  const GxfLoadExtensionsInfo info_{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  const size_t entity_num_ = 16;
  const size_t thread_num_ = 16;
  const int test_iteration_ = 10;
  bool multiThreadUpdateMultiEntity(size_t thread_num, size_t entity_num);
  size_t createTestEntities(size_t count);
  int64_t refCount(gxf_uid_t eid);
  std::vector<nvidia::gxf::Entity> test_entities_;
  gxf_tid_t tid = GxfTidNull();
};

size_t EntityThreadSafe_Test::createTestEntities(size_t count) {
  for (size_t i = 0; i < count; i++) {
    test_entities_.push_back(std::move(nvidia::gxf::Entity::New(context_).value()));
  }
  return test_entities_.size();
}

int64_t EntityThreadSafe_Test::refCount(gxf_uid_t eid) {
  int64_t count = 0;
  GXF_ASSERT_SUCCESS(GxfEntityGetRefCount(context_, eid, &count));
  return count;
}

//
// ref count thread safe tests
//
bool EntityThreadSafe_Test::multiThreadUpdateMultiEntity(size_t thread_num, size_t entity_num) {
  bool ret = true;
  // Create Entities, held by std container with ref count 1
  GXF_ASSERT_EQ(createTestEntities(entity_num), entity_num);
  // record original ref count
  std::vector<int64_t> original_ref_count;
  for (size_t i = 0; i < test_entities_.size(); ++i) {
    const gxf_uid_t eid = test_entities_.at(i).eid();
    original_ref_count.push_back(refCount(eid));
  }

  // Same number of ref Inc and Dec
  // Multi thread update multi entity ref count, imagine a matrix
  // Each row is an Entity and each column is a thread
  /*
  +------------+-----------+-----------+-----------+-----------
  |            | Thread 1  | Thread 2  | Thread 3  | Thread 4  |
  +------------+-----------+-----------+-----------+-----------
  | Entity 1   |    E11    |    E12    |    E13    |    E14    |
  +------------+-----------+-----------+-----------+-----------
  | Entity 2   |    E21    |    E22    |    E23    |    E24    |
  +------------+-----------+-----------+-----------+-----------
  | Entity 3   |    E31    |    E32    |    E33    |    E34    |
  +------------+-----------+-----------+-----------+-----------
  | Entity 4   |    E41    |    E42    |    E43    |    E44    |
  +------------+-----------+-----------+-----------+-----------
  */
  std::thread t[thread_num];
  for (size_t j = 0; j < thread_num; j++) {
    t[j] = std::thread([&, j] () mutable {
      for (size_t i = 0; i < test_entities_.size(); ++i) {
        const gxf_uid_t eid = test_entities_.at(i).eid();
        for (int k = 0; k < test_iteration_; k++) {
          GXF_ASSERT_SUCCESS(GxfEntityRefCountInc(context_, eid));
          std::string comp_name = std::string("temp_component") + std::to_string(k) + std::string("_") + std::to_string(j);
          gxf_uid_t cid = kNullUid;
          GXF_ASSERT_SUCCESS(GxfComponentAdd(context_, eid, tid, comp_name.c_str(), &cid));
        }
        for (int k = 0; k < test_iteration_; k++) {
          GXF_ASSERT_SUCCESS(GxfEntityRefCountDec(context_, eid));
          // Don't over dec otherwise it triggers auto destroy
          std::string comp_name = std::string("temp_component") + std::to_string(k) + std::string("_") + std::to_string(j);
          if (k < test_iteration_/2) {
            GXF_ASSERT_SUCCESS(ToResultCode(test_entities_.at(i).remove(tid, comp_name.c_str())));
          } else {
            GXF_ASSERT_SUCCESS(ToResultCode(test_entities_.at(i).remove<nvidia::gxf::BlockMemoryPool>(comp_name.c_str())));
          }
        }
      }
    });
  }
  for (size_t j = 0; j < thread_num; j++) {
    t[j].join();
  }

  // All threads finish, now check result
  for (size_t i = 0; i < test_entities_.size(); ++i) {
    const gxf_uid_t eid = test_entities_.at(i).eid();
    int64_t actual = refCount(eid);
    int64_t expected = original_ref_count.at(i);
    if (actual != expected) {
      GXF_LOG_ERROR("eid: %ld: actual %ld, expected %ld", eid, actual, expected);
      ret = false;
    }
  }

  test_entities_.clear();
  return ret;
}

TEST_F(EntityThreadSafe_Test, RefCountMultiThreadMultiEntity) {
  ASSERT_TRUE(multiThreadUpdateMultiEntity(thread_num_, entity_num_));
}

TEST_F(EntityThreadSafe_Test, RefCountMultiThreadSinleEntity) {
  ASSERT_TRUE(multiThreadUpdateMultiEntity(thread_num_, 1));
}

TEST_F(EntityThreadSafe_Test, RefCountSinleThreadMultiEntity) {
  ASSERT_TRUE(multiThreadUpdateMultiEntity(1, entity_num_));
}
