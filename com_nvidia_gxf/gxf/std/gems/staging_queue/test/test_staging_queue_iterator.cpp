/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/gems/staging_queue/staging_queue_iterator.hpp"

#include "gtest/gtest.h"

namespace gxf {
namespace staging_queue {

TEST(StagingQueueIterator, Basics) {
  int array[4] = {2, 5, 1, 4};
  StagingQueueIterator<int> it(array, 4, 0);
  StagingQueueIterator<int> it2 = it;
  ASSERT_EQ(it, it2);
  ASSERT_EQ(*it, 2);
  ++it;
  ASSERT_NE(it, it2);
  ASSERT_EQ(*it, 5);
  it2++;
  ASSERT_EQ(it, it2);
  ASSERT_EQ(*it2, 5);
}

TEST(StagingQueueIterator, ConstBasics) {
  int array[4] = {2, 5, 1, 4};
  StagingQueueIterator<const int> it(array, 4, 0);
  StagingQueueIterator<const int> it2 = it;
  ASSERT_EQ(it, it2);
  ASSERT_EQ(*it, 2);
  ++it;
  ASSERT_NE(it, it2);
  ASSERT_EQ(*it, 5);
  it2++;
  ASSERT_EQ(it, it2);
  ASSERT_EQ(*it2, 5);
}

TEST(StagingQueueIterator, IterateToEnd) {
  int array[4] = {2, 5, 1, 4};
  StagingQueueIterator<int> it(array, 4, 0);
  StagingQueueIterator<int> jt(array, 4, 4);
  ASSERT_NE(it, jt);
  ++it;
  ASSERT_NE(it, jt);
  ++it;
  ASSERT_NE(it, jt);
  ++it;
  ASSERT_NE(it, jt);
  ++it;
  ASSERT_EQ(it, jt);
}

TEST(StagingQueueIterator, IterateFull) {
  int array[4] = {2, 5, 1, 4};
  StagingQueueIterator<int> it1(array, 4, 0);
  StagingQueueIterator<int> it2(array, 4, 4);
  int count = 0;
  for (auto it = it1; it != it2; ++it) {
    count++;
  }
  EXPECT_EQ(count, 4);
}

TEST(StagingQueueIterator, IterateWrap) {
  int array[4] = {2, 5, 1, 4};
  StagingQueueIterator<int> it1(array, 4, 3);
  StagingQueueIterator<int> it2(array, 4, 5);
  EXPECT_EQ(*it1, 4);
  ++it1;
  EXPECT_EQ(*it1, 2);
  ++it1;
  EXPECT_EQ(it1, it2);
}

TEST(StagingQueueIterator, IterateTwice) {
  int array[4] = {2, 5, 1, 4};
  StagingQueueIterator<int> it1(array, 4, 1);
  StagingQueueIterator<int> it2(array, 4, 10);
  int count = 0;
  for (auto it = it1; it != it2; ++it) {
    count++;
  }
  EXPECT_EQ(count, 9);
}

}  // namespace staging_queue
}  // namespace gxf
