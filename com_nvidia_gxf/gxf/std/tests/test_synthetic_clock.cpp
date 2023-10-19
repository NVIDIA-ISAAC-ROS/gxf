/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gtest/gtest.h"

#include <chrono>
#include <thread>

#include "gxf/std/synthetic_clock.hpp"

namespace nvidia {
namespace gxf {
namespace test {

namespace {

constexpr const char* kExtensions[] = {
  "gxf/std/libgxf_std.so",
};

constexpr GxfLoadExtensionsInfo kExtensionInfo = { kExtensions, 1, nullptr, 0, nullptr };
constexpr GxfEntityCreateInfo kEntityInfo = {"entity", 0};

}

template <typename TComponent>
gxf::Expected<gxf::Handle<TComponent>> addComponent(gxf_context_t context,
                                                    gxf_uid_t eid, const char* name) {
  gxf_tid_t tid;
  gxf_uid_t cid;

  const gxf_result_t code1 = GxfComponentTypeId(context, TypenameAsString<TComponent>(), &tid);
  if (code1 != GXF_SUCCESS) {
    return gxf::Unexpected{code1};
  }

  const gxf_result_t code2 = GxfComponentAdd(context, eid, tid, name, &cid);

  if (code2 != GXF_SUCCESS) {
    return gxf::Unexpected{code2};
  }

  return gxf::Handle<TComponent>::Create(context, cid);
}

class TestSyntheticClock : public ::testing::Test {
 protected:
  void SetUp() {
    ASSERT_EQ(GxfContextCreate(&context_), GXF_SUCCESS);
    ASSERT_EQ(GxfLoadExtensions(context_, &kExtensionInfo), GXF_SUCCESS);
    ASSERT_EQ(GxfCreateEntity(context_, &kEntityInfo, &eid_), GXF_SUCCESS);

    ASSERT_TRUE(addComponent<gxf::SyntheticClock>(context_, eid_, "synthetic_clock")
      .assign_to(synthetic_clock_));

    ASSERT_EQ(synthetic_clock_->initialize(), GXF_SUCCESS);
  }

  void TearDown() {
    ASSERT_EQ(synthetic_clock_->deinitialize(), GXF_SUCCESS);
    ASSERT_EQ(GxfEntityDestroy(context_, eid_), GXF_SUCCESS);
    ASSERT_EQ(GxfContextDestroy(context_), GXF_SUCCESS);
  }

  gxf_context_t context_;
  gxf_uid_t eid_;
  gxf::Handle<gxf::SyntheticClock> synthetic_clock_;
};

TEST_F(TestSyntheticClock, Instantiate) {
  // do nothing, just check whether the fixture works
}

TEST_F(TestSyntheticClock, SimpleAdvance) {
  ASSERT_EQ(synthetic_clock_->timestamp(), 0);
  ASSERT_TRUE(synthetic_clock_->advanceTo(1000));
  ASSERT_EQ(synthetic_clock_->timestamp(), 1000);
}

TEST_F(TestSyntheticClock, ParallelAdvance) {
  std::mutex mutex;
  bool has_advanced = false;

  ASSERT_TRUE(synthetic_clock_->advanceTo(23));

  std::thread waiter_thread([&] {
    while (true) {
      std::lock_guard<std::mutex> guard(mutex);

      if (has_advanced) {
        ASSERT_EQ(synthetic_clock_->timestamp(), 42);
        return;
      }

      ASSERT_FALSE(has_advanced);
      ASSERT_EQ(synthetic_clock_->timestamp(), 23);
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  {
    std::lock_guard<std::mutex> guard(mutex);

    ASSERT_TRUE(synthetic_clock_->advanceTo(42));
    has_advanced = true;
  }

  waiter_thread.join();
}

TEST_F(TestSyntheticClock, CompressesTime) {
  std::thread waiter_thread([&] {
    ASSERT_TRUE(synthetic_clock_->sleepUntil(100 * 1000)); // sleep for 100ms
    ASSERT_EQ(synthetic_clock_->timestamp(), 100 * 1000);
  });

  std::this_thread::sleep_for(std::chrono::nanoseconds(10 * 1000)); // only 10ms wall clock
  ASSERT_TRUE(synthetic_clock_->advanceBy(100 * 1000));

  waiter_thread.join();
}

TEST_F(TestSyntheticClock, Overshooting) {
  constexpr int64_t kTargetTime = 123456789;

  std::thread waiter_thread([&] {
    ASSERT_TRUE(synthetic_clock_->sleepUntil(kTargetTime));
    ASSERT_GE(synthetic_clock_->timestamp(), kTargetTime);
  });

  std::this_thread::sleep_for(std::chrono::nanoseconds(kTargetTime));
  ASSERT_TRUE(synthetic_clock_->advanceBy(kTargetTime * 5));

  waiter_thread.join();
}

// make sure sleeping works if time was set before
TEST_F(TestSyntheticClock, AlreadyAdvanced) {
  constexpr int64_t kTargetTime = 123456789;
  ASSERT_TRUE(synthetic_clock_->advanceBy(kTargetTime * 5));

  std::thread waiter_thread([&] {
    ASSERT_TRUE(synthetic_clock_->sleepUntil(kTargetTime));
    ASSERT_GE(synthetic_clock_->timestamp(), kTargetTime);
  });

  waiter_thread.join();
}

TEST_F(TestSyntheticClock, AdvanceRaceCondition) {
  constexpr int64_t kTargetTime = 123456789;

  std::thread waiter_thread([&] {
    ASSERT_TRUE(synthetic_clock_->sleepUntil(kTargetTime));
    ASSERT_GE(synthetic_clock_->timestamp(), kTargetTime);
  });

  // advance immediately
  ASSERT_TRUE(synthetic_clock_->advanceBy(kTargetTime * 5));

  waiter_thread.join();
}

}  // namespace test
}  // namespace gxf
}  // namespace nvidia
