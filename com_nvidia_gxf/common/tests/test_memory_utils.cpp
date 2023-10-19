/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "common/memory_utils.hpp"

#include "gtest/gtest.h"

TEST(MemoryUtils, DefaultConstruction) {
  static uint8_t construct = 0;
  construct = 0;

  struct A{
    A() : x_(0), y_(0) { ++construct; }
    int x_;
    int y_;
  };

  byte buffer[sizeof(A)] = {0};
  nvidia::InplaceConstruct<A>(buffer);
  EXPECT_TRUE(nvidia::ValuePointer<A>(buffer)->x_ == 0);
  EXPECT_TRUE(nvidia::ValuePointer<A>(buffer)->y_ == 0);
  EXPECT_TRUE(construct == 1);
}

TEST(MemoryUtils, ArgumentConstruction) {
  static uint8_t construct;
  construct = 0;

  struct A{
    A(int x, int y)   : x_(x), y_(y) { ++construct; }
    int x_;
    int y_;
  };


  byte buffer[sizeof(A)] = {0};
  nvidia::InplaceConstruct<A>(buffer, 1, 2);
  EXPECT_TRUE(nvidia::ValuePointer<A>(buffer)->x_ == 1);
  EXPECT_TRUE(nvidia::ValuePointer<A>(buffer)->y_ == 2);
  EXPECT_TRUE(construct == 1);
}

TEST(MemoryUtils, CopyConstruction) {
  static uint8_t copy;
  copy = 0;

  struct A{
    A(int x, int y)   : x_(x), y_(y) {}
    A(const A& other) : x_(other.x_), y_(other.y_) { ++copy; }
    int x_;
    int y_;
  };

  byte buffer[sizeof(A)] = {0};
  A a(5,6);
  nvidia::InplaceCopyConstruct(buffer, a);
  EXPECT_TRUE(nvidia::ValuePointer<A>(buffer)->x_ == 5);
  EXPECT_TRUE(nvidia::ValuePointer<A>(buffer)->y_ == 6);
  EXPECT_TRUE(copy == 1);
}

TEST(MemoryUtils, MoveConstruction) {
  static uint8_t move;
  move = 0;

  struct A{
    A(int x, int y)   : x_(x), y_(y) {}
    A(A&& other) : x_(other.x_), y_(other.y_) { ++move; }
    int x_;
    int y_;
  };

  byte buffer[sizeof(A)] = {0};
  nvidia::InplaceMoveConstruct(buffer, A(8,9));
  EXPECT_TRUE(nvidia::ValuePointer<A>(buffer)->x_ == 8);
  EXPECT_TRUE(nvidia::ValuePointer<A>(buffer)->y_ == 9);
  EXPECT_TRUE(move == 1);
}

TEST(MemoryUtils, Destruction) {
  static uint8_t destruct;
  destruct = 0;

  struct A{
    A() {}
    ~A() { ++destruct; }
  };

  byte buffer[sizeof(A)] = {0};
  nvidia::InplaceConstruct<A>(buffer);
  nvidia::Destruct<A>(buffer);
  EXPECT_TRUE(destruct == 1);
}
