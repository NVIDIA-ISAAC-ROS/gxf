/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/gems/suballocators/first_fit_allocator_base.hpp"

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

namespace nvidia {
namespace gxf {
namespace {
constexpr int kBufferSize = 1010101;
constexpr int kMaxSize = 10101;
}  // namespace

TEST(FirstFitAllocatorBase, NotInitialized) {
  FirstFitAllocatorBase management;
  ASSERT_FALSE(management.acquire(32));
  ASSERT_FALSE(management.release(0));
}

TEST(FirstFitAllocatorBase, allocate) {
  FirstFitAllocatorBase management;
  ASSERT_FALSE(management.allocate(-1));
  ASSERT_FALSE(management.allocate(0));
  ASSERT_FALSE(management.allocate(1<<30));
  ASSERT_TRUE(management.allocate(1<<20));
  // Reallocation is allowed only if no memory is used
  auto idx = management.acquire(42);
  ASSERT_TRUE(idx);
  ASSERT_FALSE(management.allocate(1<<10));
  ASSERT_TRUE(management.release(idx.value()));
  ASSERT_TRUE(management.allocate(1<<10));
}

TEST(FirstFitAllocatorBase, acquire) {
  FirstFitAllocatorBase management;
  int size = kBufferSize;
  ASSERT_TRUE(management.allocate(size));
  // Acquire all the available memory until it's full
  while (size) {
    int ss = rand()%std::min(size, 10101) + 1;
    size -= ss;
    ASSERT_TRUE(management.acquire(ss));
  }
  // Next query should fail as no more memory is available
  ASSERT_FALSE(management.acquire(1));
}

TEST(FirstFitAllocatorBase, release) {
  FirstFitAllocatorBase management;
  int size = kBufferSize;
  ASSERT_TRUE(management.allocate(size));
  std::vector<int> blocks;
  // Fill the memory and store the memory blocks
  while (size) {
    int ss = rand()%std::min(size, kMaxSize) + 1;
    size -= ss;
    blocks.push_back(management.acquire(ss).value());
  }
  // Check we can't release a block not acquired.
  for (size_t idx = 1; idx < blocks.size(); idx++) {
    for (int pos = blocks[idx-1] + 1; pos < blocks[idx]; pos++) {
      ASSERT_FALSE(management.release(pos)) << pos;
    }
  }
  // check that we can release all the blocks in a random order, and once released, it can't be
  // released again.
  std::random_shuffle(blocks.begin(), blocks.end());
  for (const auto& block : blocks) {
    ASSERT_TRUE(management.release(block));
    ASSERT_FALSE(management.release(block));
  }
}

TEST(FirstFitAllocatorBase, RandomQuery) {
  FirstFitAllocatorBase management;
  int size = kBufferSize;
  ASSERT_TRUE(management.allocate(size));
  // Helper to manually check no memory is acquired twice, and failure were indeed correct.
  std::vector<int> mem(size, 0);
  std::vector<std::pair<int, int>> blocks;
  for (int it = 1; it <= 128 * 1024; it++) {
    // We do 2x attempt to allocate memory to be sure the memory end up full.
    if (rand()%3) {
      int ss = rand() % kMaxSize + 1;
      auto index = management.acquire(ss);
      if (index) {
        // Acquisition succeed, let's make sure the returned range was valid.
        for (int i = 0; i < ss; i++) {
          ASSERT_EQ(mem[i + index.value()], 0);
          mem[i + index.value()] = it;
        }
        blocks.push_back({index.value(), ss});
      } else {
        // Acquisition failed, let's go through the full memory block and check there were no block
        // of the required size.
        // In order to speed up the process, we first move the pointer by ss, and we look backward.
        // If we hit an acquired cell, we know that the next available block has to start with the
        // next index or higher. So we jump again by ss.
        int idx = ss-1;
        while (idx < size) {
          bool valid = true;
          for (int i = 0; i < ss; i++, idx--) {
            if (mem[idx] != 0) {
              valid = false;
              idx += ss;
              break;
            }
          }
          ASSERT_FALSE(valid);
        }
      }
    } else {
      // Sanity check, this is unlikely to happen after few iterations but could be the case in the
      // first few iterations.
      if (blocks.empty()) continue;
      int r = rand() % blocks.size();
      // Free the memory
      for (int i = 0; i < blocks[r].second; i++) mem[i + blocks[r].first] = 0;
      ASSERT_TRUE(management.release(blocks[r].first));
      blocks[r] = blocks.back();
      blocks.pop_back();
    }
  }
}

}  // namespace gxf
}  // namespace nvidia
