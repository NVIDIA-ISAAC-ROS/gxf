/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/gems/staging_queue/staging_queue.hpp"

#include <algorithm>
#include <random>
#include <thread>

#include "gtest/gtest.h"

namespace gxf {
namespace staging_queue {

// Checks that the queue contains exactly the expected elements
template <typename T>
void CheckQueue(const StagingQueue<T>& queue, const std::vector<T>& expected) {
  ASSERT_EQ(queue.size(), expected.size());
  if (expected.size() > 0) {
    ASSERT_EQ(queue.peek(), expected.front());
    ASSERT_EQ(queue.latest(), expected.back());
  }
  for (size_t i = 0; i < queue.size(); i++) {
    ASSERT_EQ(queue.peek(i), expected[i]);
    ASSERT_EQ(queue.latest(i), expected[expected.size() - 1 - i]);
  }
}

TEST(StagingQueue, Count) {
  StagingQueue<int> queue(2, OverflowBehavior::kPop, 0);
  EXPECT_EQ(queue.size(), 0);
  queue.push(1);
  EXPECT_EQ(queue.size(), 0);
  queue.push(1);
  EXPECT_EQ(queue.size(), 0);
  queue.sync();
  EXPECT_EQ(queue.size(), 2);
  queue.push(1);
  EXPECT_EQ(queue.size(), 2);
  queue.push(1);
  EXPECT_EQ(queue.size(), 2);
  queue.sync();
  EXPECT_EQ(queue.size(), 2);
  EXPECT_EQ(queue.pop(), 1);
  EXPECT_EQ(queue.size(), 1);
  EXPECT_EQ(queue.pop(), 1);
  EXPECT_EQ(queue.size(), 0);
  queue.sync();
  EXPECT_EQ(queue.size(), 0);
}

TEST(StagingQueue, CountThorough) {
  std::mt19937 rng;
  std::uniform_int_distribution<int> rnd_capacity(1, 5);
  std::uniform_int_distribution<int> rnd_pushs(0, 10);
  std::uniform_int_distribution<int> rnd_pops(0, 10);
  for (int k = 0; k < 100; k++) {
    const int capacity = rnd_capacity(rng);
    StagingQueue<int> queue(capacity, OverflowBehavior::kPop, 0);
    ASSERT_EQ(queue.size(), 0);
    int num_main = 0;
    int num_back = 0;
    for (int i = 0; i < 100; i++) {
      // Push some elements
      const int npush = rnd_pushs(rng);
      for (int j = 0; j < npush; j++) {
        queue.push(0);
        num_back = std::min(num_back + 1, capacity);
        ASSERT_EQ(queue.size(), num_main);
      }
      // Sync
      queue.sync();
      num_main = std::min(num_main + num_back, capacity);
      num_back = 0;
      ASSERT_EQ(queue.size(), num_main);
      // Pop some elements
      const int npop = rnd_pops(rng);
      for (int j = 0; j < npop; j++) {
        queue.pop();
        num_main = std::max(num_main - 1, 0);
        ASSERT_EQ(queue.size(), num_main);
      }
    }
  }
}

TEST(StagingQueue, PopOverflowBackstage) {
  StagingQueue<int> queue(2, OverflowBehavior::kPop, -1);
  queue.push(1);
  queue.push(2);
  queue.sync();
  queue.push(3);
  queue.push(4);
  queue.push(5);
  queue.sync();
  ASSERT_EQ(queue.size(), 2);
  EXPECT_EQ(queue.peek(), 4);
  EXPECT_EQ(queue.peek(1), 5);
}

TEST(StagingQueue, PushOverflowPopBasics) {
  StagingQueue<int> queue(2, OverflowBehavior::kPop, -1);
  queue.push(0);
  queue.push(1);
  queue.push(7);
  EXPECT_EQ(queue.size(), 0);
  queue.sync();
  EXPECT_EQ(queue.size(), 2);
  EXPECT_EQ(queue.peek(), 1);
  EXPECT_EQ(queue.peek(0), 1);
  EXPECT_EQ(queue.peek(1), 7);
  EXPECT_EQ(queue.latest(), 7);
  EXPECT_EQ(queue.latest(0), 7);
  EXPECT_EQ(queue.latest(1), 1);
  queue.push(13);
  EXPECT_EQ(queue.size(), 2);
  queue.sync();
  EXPECT_EQ(queue.size(), 2);
  EXPECT_EQ(queue.peek(), 7);
  EXPECT_EQ(queue.peek(0), 7);
  EXPECT_EQ(queue.peek(1), 13);
  EXPECT_EQ(queue.latest(), 13);
  EXPECT_EQ(queue.latest(0), 13);
  EXPECT_EQ(queue.latest(1), 7);
  queue.push(3);
  EXPECT_EQ(queue.size(), 2);
  queue.sync();
  EXPECT_EQ(queue.size(), 2);
  EXPECT_EQ(queue.peek(), 13);
  EXPECT_EQ(queue.peek(0), 13);
  EXPECT_EQ(queue.peek(1), 3);
  EXPECT_EQ(queue.latest(), 3);
  EXPECT_EQ(queue.latest(0), 3);
  EXPECT_EQ(queue.latest(1), 13);
}

TEST(StagingQueue, PushOverflowPop) {
  StagingQueue<int> queue(3, OverflowBehavior::kPop, -1);
  queue.push(0);
  queue.push(1);
  queue.push(2);
  queue.push(3);
  CheckQueue(queue, {});
  queue.sync();
  CheckQueue(queue, {1, 2, 3});
  queue.push(12);
  queue.push(13);
  queue.sync();
  CheckQueue(queue, {3, 12, 13});
}

TEST(StagingQueue, PushOverflowReject) {
  StagingQueue<int> queue(3, OverflowBehavior::kReject, -1);
  queue.push(0);
  queue.push(1);
  queue.push(2);
  queue.push(3);
  CheckQueue(queue, {});
  queue.sync();
  CheckQueue(queue, {0, 1, 2});
  queue.push(12);
  queue.push(13);
  queue.sync();
  CheckQueue(queue, {0, 1, 2});
}

TEST(StagingQueue, PushOverflowFault1) {
  StagingQueue<int> queue(3, OverflowBehavior::kFault, -1);
  queue.push(0);
  queue.push(1);
  queue.push(2);
  EXPECT_FALSE(queue.push(3));
}

TEST(StagingQueue, PushOverflowFault2) {
  StagingQueue<int> queue(3, OverflowBehavior::kFault, -1);
  queue.push(0);
  queue.push(1);
  queue.push(2);
  queue.sync();
  CheckQueue(queue, {0, 1, 2});
  queue.push(3);
  EXPECT_FALSE(queue.sync());
}

TEST(StagingQueue, PushOverflowFault3a) {
  StagingQueue<int> queue(3, OverflowBehavior::kFault, -1);
  queue.push(0);
  queue.push(1);
  queue.push(2);
  queue.sync();
  CheckQueue(queue, {0, 1, 2});
  queue.push(3);
  queue.pop();
  queue.sync();
  CheckQueue(queue, {1, 2, 3});
}

TEST(StagingQueue, PushOverflowFault3b) {
  StagingQueue<int> queue(3, OverflowBehavior::kFault, -1);
  queue.push(0);
  queue.push(1);
  queue.push(2);
  queue.sync();
  CheckQueue(queue, {0, 1, 2});
  queue.push(3);
  queue.push(4);
  queue.push(5);
  queue.pop();
  queue.pop();
  queue.pop();
  queue.sync();
  CheckQueue(queue, {3, 4, 5});
}

TEST(StagingQueue, PushOverflowFault3c) {
  StagingQueue<int> queue(3, OverflowBehavior::kFault, -1);
  queue.push(0);
  queue.push(1);
  queue.push(2);
  queue.sync();
  CheckQueue(queue, {0, 1, 2});
  queue.push(3);
  queue.push(4);
  queue.push(5);
  EXPECT_FALSE(queue.push(6));
}

TEST(StagingQueue, Pop) {
  StagingQueue<int> queue(3, OverflowBehavior::kPop, -1);
  queue.push(0);
  queue.push(1);
  queue.push(2);
  queue.push(3);
  EXPECT_EQ(queue.size(), 0);
  queue.sync();
  CheckQueue(queue, {1, 2, 3});
  EXPECT_EQ(queue.pop(), 1);
  CheckQueue(queue, {2, 3});
  EXPECT_EQ(queue.pop(), 2);
  CheckQueue(queue, {3});
  EXPECT_EQ(queue.pop(), 3);
  CheckQueue(queue, {});
}

TEST(StagingQueue, Concurrency) {
  StagingQueue<int> queue(100, OverflowBehavior::kPop, -1);
  std::thread t1([&] {
    for (int k = 0; k < 10; k++) {
      while (!queue.empty())
        ;
      for (int i = 0; i < 10; i++) {
        queue.push(i);
      }
      queue.sync();
    }
    queue.push(1337);
    queue.sync();
  });
  std::thread t2([&] {
    for (int k = 0; k < 10; k++) {
      for (int i = 0; i < 10; i++) {
        while (queue.empty())
          ;
        ASSERT_EQ(queue.peek(), i);
        EXPECT_EQ(queue.pop(), i);
      }
    }
  });
  t1.join();
  t2.join();
  CheckQueue(queue, {1337});
}

TEST(StagingQueue, UseCase1_PopLoop) {
  StagingQueue<int> queue(5, OverflowBehavior::kPop, -1);
  queue.push(1);
  queue.push(1);
  queue.push(2);
  queue.push(3);
  queue.sync();

  while (!queue.empty()) {
    if (queue.peek() == 1) {
      queue.pop();
    } else {
      return;
    }
  }

  ASSERT_EQ(queue.size(), 2);
  EXPECT_EQ(queue.peek(0), 2);
  EXPECT_EQ(queue.peek(1), 3);
}

TEST(StagingQueue, UseCase2_Peek) {
  StagingQueue<int> queue(5, OverflowBehavior::kPop, -1);
  queue.push(1);
  queue.push(1);
  queue.push(2);
  queue.push(3);
  queue.sync();

  std::vector<int> buffer(queue.size());
  for (size_t i = 0; i < queue.size(); i++) {
    buffer[i] = 2 * queue.peek(i);
  }

  ASSERT_EQ(buffer.size(), 4);
  EXPECT_EQ(buffer[0], 2);
  EXPECT_EQ(buffer[1], 2);
  EXPECT_EQ(buffer[2], 4);
  EXPECT_EQ(buffer[3], 6);
}

TEST(StagingQueue, UseCase3_Latest) {
  StagingQueue<double> queue(2, OverflowBehavior::kPop, 0.0);
  queue.push(0.1);
  queue.push(0.4);
  queue.push(1.2);
  queue.sync();

  ASSERT_EQ(queue.size(), 2);
  const double actual = 0.7 * queue.latest() + 0.3 * queue.latest(1);

  EXPECT_NEAR(actual, 0.96, 1e-9);
}

TEST(StagingQueue, UseCase4_StlCompatibility) {
  StagingQueue<int> queue(2, OverflowBehavior::kPop, 0);
  queue.push(101);
  queue.push(104);
  queue.push(112);
  queue.sync();

  std::vector<int> other(queue.size());
  ASSERT_EQ(std::distance(queue.begin(), queue.end()), queue.size());
  std::copy(queue.begin(), queue.end(), other.begin());
  std::copy(std::begin(queue), std::end(queue), other.begin());

  ASSERT_EQ(other.size(), 2);
  ASSERT_EQ(other[0], 104);
  ASSERT_EQ(other[1], 112);
}

TEST(StagingQueue, UseCase5_RangeBasedFor) {
  StagingQueue<int> queue(25, OverflowBehavior::kPop, -1);
  for (int i = 0; i < 40; i++) {
    queue.push(i);
  }
  queue.sync();

  ASSERT_EQ(queue.size(), 25);
  int i = 15;
  for (int x : queue) {
    ASSERT_EQ(x, i++);
  }
}

TEST(StagingQueue, ContinuousPopOverflow) {
  StagingQueue<int> queue(3, OverflowBehavior::kPop, -1);
  queue.push(1);
  queue.push(2);
  queue.push(3);
  queue.sync();

  for (int i = 4; i < 30;) {
    for (int j = 0; j < 3; j++) {
      queue.push(i++);
    }
    queue.sync();
    ASSERT_EQ(queue.size(), 3);
    int j = i - 3;
    for (size_t k = 0; k < queue.size(); k++) {
      EXPECT_EQ(queue.peek(k), j++);
    }
  }
}

TEST(StagingQueue, Iterator) {
  StagingQueue<int> queue(3, OverflowBehavior::kPop, 0);
  ASSERT_EQ(queue.begin(), queue.end());
  ASSERT_EQ(std::distance(queue.begin(), queue.end()), 0);

  queue.push(1);
  ASSERT_EQ(queue.begin(), queue.end());
  ASSERT_EQ(std::distance(queue.begin(), queue.end()), 0);

  queue.push(2);
  ASSERT_EQ(queue.begin(), queue.end());
  ASSERT_EQ(std::distance(queue.begin(), queue.end()), 0);

  queue.push(3);
  ASSERT_EQ(queue.begin(), queue.end());
  ASSERT_EQ(std::distance(queue.begin(), queue.end()), 0);

  queue.sync();
  ASSERT_NE(queue.begin(), queue.end());
  ASSERT_EQ(std::distance(queue.begin(), queue.end()), 3);

  for (int i = 4; i < 100;) {
    for (int j = 0; j < 5; j++) {
      queue.push(i++);
    }
    queue.sync();
    ASSERT_EQ(queue.size(), 3);
    ASSERT_EQ(std::distance(queue.begin(), queue.end()), 3);
    int j = i - 3;
    for (auto it = queue.begin(); it != queue.end(); ++it) {
      ASSERT_EQ(*it, j++);
    }
  }
}

}  // namespace staging_queue
}  // namespace gxf
