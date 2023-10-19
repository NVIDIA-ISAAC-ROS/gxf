/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gtest/gtest.h"

#include <cstdint>

#include "common/strong_type.hpp"

namespace nvidia {
namespace test {

namespace {

using Foo = StrongType<struct foo_t, uint64_t>;
using Bar = StrongType<struct bar_t, uint64_t>;

uint64_t fooOnly(const Foo& foo) { return foo.value(); }
uint64_t barOnly(const Bar& bar) { return bar.value(); }

template <typename, typename = void>
struct IsFooCallable : FalseType {};

template <typename T>
struct IsFooCallable<T, void_t<decltype(fooOnly(Declval<T>()))>> : TrueType {};

struct StructType {
  char letter;
  bool flag;
};

using StrongStructType = StrongType<struct struct_type_t, StructType>;

static_assert(IsNothrowDefaultConstructible<Foo>::value,
              "Should be no-throw default-constructible because uint64_t is");
static_assert(IsNothrowConstructible<Foo, uint64_t>::value,
              "Should be no-throw constructible from value type");
static_assert(!IsFooCallable<uint64_t>::value, "fooOnly() should not be callable with uint64_t");
static_assert(!IsFooCallable<Bar>::value, "fooOnly() should not beis callable with Bar");
static_assert(IsFooCallable<Foo>::value, "fooOnly() should be callable with Foo");

}  // namespace

TEST(TestStrongType, FooBar) {
  Foo foo = Foo(106);
  Bar bar = Bar(314);
  EXPECT_EQ(static_cast<uint64_t>(foo), 106);
  EXPECT_EQ(static_cast<uint64_t>(bar), 314);
  EXPECT_EQ(fooOnly(foo), 106);
  EXPECT_EQ(barOnly(bar), 314);
}

TEST(TestStrongType, Struct) {
  StrongStructType a = StrongStructType(StructType{'a', true});
  StrongStructType b = StrongStructType(StructType{'b', false});
  EXPECT_EQ(a->letter, 'a');
  EXPECT_EQ(b->letter, 'b');
  EXPECT_TRUE(a->flag);
  EXPECT_FALSE(b->flag);
}

TEST(TestStrongType, Comparison) {
  Foo a(99);
  Foo b(99);
  Foo c(200);

  // operator==
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a == c);

  // operator!=
  EXPECT_FALSE(a != b);
  EXPECT_TRUE(a != c);

  // operator<=
  EXPECT_TRUE(a <= b);
  EXPECT_FALSE(c <= a);
  // operator>=
  EXPECT_TRUE(a >= b);
  EXPECT_FALSE(a >= c);

  // operator>
  EXPECT_FALSE(a > b);
  EXPECT_TRUE(c > a);

  // operator<
  EXPECT_FALSE(a < b);
  EXPECT_TRUE(a < c);
}

}  // namespace test
}  // namespace nvidia

