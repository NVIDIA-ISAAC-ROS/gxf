/*
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/scheduling_terms.hpp"

#include "gxf/std/block_memory_pool.hpp"

#include "gtest/gtest.h"

namespace nvidia {
namespace gxf {

constexpr uint64_t kPoolSize = 8UL;
constexpr uint64_t kBlocksRequested = 2UL;
constexpr uint64_t kBlockSize = 128UL; // not relevant in this test

class TestMemoryAvailableSchedulingTerm : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_EQ(GxfContextCreate(&context_), GXF_SUCCESS);

    constexpr const char* kExtensions[] = {
      "gxf/std/libgxf_std.so",
    };
    const GxfLoadExtensionsInfo info{kExtensions, 1, nullptr, 0, nullptr};

    ASSERT_EQ(GxfLoadExtensions(context_, &info), GXF_SUCCESS);

    gxf_uid_t eid;
    const GxfEntityCreateInfo entity_create_info = {0};
    ASSERT_EQ(GxfCreateEntity(context_, &entity_create_info, &eid), GXF_SUCCESS);

    gxf_tid_t pool_tid;
    ASSERT_EQ(GxfComponentTypeId(context_, "nvidia::gxf::BlockMemoryPool", &pool_tid), GXF_SUCCESS);
    gxf_uid_t pool_cid;
    ASSERT_EQ(GxfComponentAdd(context_, eid, pool_tid, "term", &pool_cid), GXF_SUCCESS);

    gxf_tid_t term_tid;
    ASSERT_EQ(GxfComponentTypeId(context_, "nvidia::gxf::MemoryAvailableSchedulingTerm", &term_tid), GXF_SUCCESS);
    gxf_uid_t term_cid;
    ASSERT_EQ(GxfComponentAdd(context_, eid, term_tid, "term", &term_cid), GXF_SUCCESS);

    ASSERT_EQ(GxfParameterSetUInt64(context_, pool_cid, "block_size", kBlockSize), GXF_SUCCESS);
    ASSERT_EQ(GxfParameterSetUInt64(context_, pool_cid, "num_blocks", kPoolSize), GXF_SUCCESS);

    ASSERT_EQ(GxfParameterSetHandle(context_, term_cid, "allocator", pool_cid), GXF_SUCCESS);
    ASSERT_EQ(GxfParameterSetUInt64(context_, term_cid, "min_blocks", kBlocksRequested), GXF_SUCCESS);

    void* poolPointer;
    ASSERT_EQ(GxfComponentPointer(context_, pool_cid, pool_tid, &poolPointer), GXF_SUCCESS);
    pool_ = static_cast<BlockMemoryPool*>(poolPointer);

    void* termPointer;
    ASSERT_EQ(GxfComponentPointer(context_, term_cid, term_tid, &termPointer), GXF_SUCCESS);
    term_ = static_cast<MemoryAvailableSchedulingTerm*>(termPointer);

    ASSERT_EQ(pool_->initialize(), GXF_SUCCESS);
    ASSERT_EQ(term_->initialize(), GXF_SUCCESS);
  }

  void TearDown() override {
    ASSERT_EQ(term_->deinitialize(), GXF_SUCCESS);
    ASSERT_EQ(pool_->deinitialize(), GXF_SUCCESS);

    ASSERT_EQ(GxfContextDestroy(context_), GXF_SUCCESS);
  }

  gxf_context_t context_;
  BlockMemoryPool* pool_;
  MemoryAvailableSchedulingTerm* term_;
};

TEST_F(TestMemoryAvailableSchedulingTerm, canCreate) {
  // do nothing, just make sure we can instantiate everything
}

TEST_F(TestMemoryAvailableSchedulingTerm, waitAfterAllocate) {
  SchedulingConditionType type;
  int64_t now = 0;
  int64_t time_updated;

  ASSERT_EQ(term_->check_abi(now, &type, &time_updated), GXF_SUCCESS);
  // if we haven't checked, we should not start
  ASSERT_EQ(type, SchedulingConditionType::WAIT);
  ASSERT_EQ(time_updated, 0);

  ++now; // advance time

  ASSERT_EQ(term_->update_state_abi(now), GXF_SUCCESS);
  ASSERT_EQ(term_->check_abi(now, &type, &time_updated), GXF_SUCCESS);
  ASSERT_EQ(type, SchedulingConditionType::READY);
  ASSERT_EQ(time_updated, now);

  // should not do anything as long as we have enough space
  for (uint64_t i = 0; i < (kPoolSize - kBlocksRequested); ++i) {
    ++now; // advance our "time"

    void* ptr; // we just ignore those
    ASSERT_EQ(pool_->allocate_abi(kBlockSize, 0, &ptr), GXF_SUCCESS);

    ASSERT_EQ(term_->update_state_abi(now), GXF_SUCCESS);
    ASSERT_EQ(term_->check_abi(now, &type, &time_updated), GXF_SUCCESS);
    ASSERT_EQ(type, SchedulingConditionType::READY); // should still have space
    ASSERT_EQ(time_updated, 1); // should not have been updated since beginning
  }

  ++now; // advance once more

  void* ptr; // not used
  ASSERT_EQ(pool_->allocate_abi(kBlockSize, 0, &ptr), GXF_SUCCESS);

  ASSERT_EQ(term_->update_state_abi(now), GXF_SUCCESS);
  ASSERT_EQ(term_->check_abi(now, &type, &time_updated), GXF_SUCCESS);
  ASSERT_EQ(type, SchedulingConditionType::WAIT); // now we should be waiting
  ASSERT_EQ(time_updated, now); // should have updated on this allocation

  // should stay in waiting state from now on
  for (uint64_t i = 0; i < kBlocksRequested - 1; ++i) {
    void* ptr; // we just ignore those
    ASSERT_EQ(pool_->allocate_abi(kBlockSize, 0, &ptr), GXF_SUCCESS);

    ASSERT_EQ(term_->update_state_abi(now), GXF_SUCCESS);
    ASSERT_EQ(term_->check_abi(now + i + 1, &type, &time_updated), GXF_SUCCESS);
    ASSERT_EQ(type, SchedulingConditionType::WAIT); // should not have space anymore
    ASSERT_EQ(time_updated, now); // should not have been updated since it flipped
  }
}

TEST_F(TestMemoryAvailableSchedulingTerm, readyAfterFree) {
  SchedulingConditionType type;
  int64_t now = 1;
  int64_t time_updated;

  void* ptrs[kPoolSize];

  // drani the pool first
  for (uint64_t i = 0; i < kPoolSize; ++i) {
    ASSERT_EQ(pool_->allocate_abi(kBlockSize, 0, &ptrs[i]), GXF_SUCCESS);
  }

  ASSERT_EQ(term_->update_state_abi(now), GXF_SUCCESS);
  ASSERT_EQ(term_->check_abi(now, &type, &time_updated), GXF_SUCCESS);
  ASSERT_EQ(type, SchedulingConditionType::WAIT); // pool is full, should be waiting
  ASSERT_EQ(time_updated, 0); // should still be in wait from beginning

  // free until one before the switch
  for (uint64_t i = 0; i < kBlocksRequested - 1; ++i) {
    ++now; // advance time

    ASSERT_EQ(pool_->free_abi(ptrs[i]), GXF_SUCCESS);

    ASSERT_EQ(term_->update_state_abi(now), GXF_SUCCESS);
    ASSERT_EQ(term_->check_abi(now, &type, &time_updated), GXF_SUCCESS);
    ASSERT_EQ(type, SchedulingConditionType::WAIT); // pool is full, should be waiting
    ASSERT_EQ(time_updated, 0); // should still be in wait from beginning
  }

  ++now;
  int64_t time_freed = now;

  ASSERT_EQ(pool_->free_abi(ptrs[kBlocksRequested - 1]), GXF_SUCCESS);

  ASSERT_EQ(term_->update_state_abi(now), GXF_SUCCESS);
  ASSERT_EQ(term_->check_abi(now, &type, &time_updated), GXF_SUCCESS);
  ASSERT_EQ(type, SchedulingConditionType::READY); // should have just enough space
  ASSERT_EQ(time_updated, now); // should have updated right now

  // drain the rest of the pool, should not change anything
  for (uint64_t i = 0; i < kPoolSize - kBlocksRequested; ++i) {
    ++now; // advance time

    ASSERT_EQ(pool_->free_abi(ptrs[kBlocksRequested + i]), GXF_SUCCESS) << i;

    ASSERT_EQ(term_->update_state_abi(now), GXF_SUCCESS);
    ASSERT_EQ(term_->check_abi(now, &type, &time_updated), GXF_SUCCESS);
    ASSERT_EQ(type, SchedulingConditionType::READY); // should still have enough space
    ASSERT_EQ(time_updated, time_freed); // should not have changed since flip
  }
}

}  // namespace gxf
}  // namespace nvidia
