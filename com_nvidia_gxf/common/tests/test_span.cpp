/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gtest/gtest.h"

#include <numeric>
#include <vector>

#include "common/span.hpp"

namespace nvidia {
namespace test {

template <typename T>
class TestSpanTyped : public testing::TestWithParam<T> {};

using PrimitiveTypes = ::testing::Types<bool,
                                        float, double,
                                        int8_t, int16_t, int32_t, int64_t,
                                        uint8_t, uint16_t, uint32_t, uint64_t>;

TYPED_TEST_CASE(TestSpanTyped, PrimitiveTypes);

TYPED_TEST(TestSpanTyped, Empty) {
  using T = TypeParam;
  constexpr size_t count = 256;

  // create Span holding unmanaged pointer
  T* ptr = new T[count];

  // zero-argument constructor
  Span<T> empty;
  EXPECT_EQ(empty.size(), 0);
  EXPECT_EQ(empty.data(), nullptr);

  empty = Span<T>(ptr, count);
  EXPECT_EQ(empty.size(), count);
  EXPECT_EQ(empty.data(), ptr);

  delete[] ptr;
}

TYPED_TEST(TestSpanTyped, Ops) {
  using T = TypeParam;

  for (size_t count : {1, 256, 256 * 256}) {
    // create Span holding unmanaged pointer
    T* ptr = new T[count];
    Span<T> span(ptr, count);
    const Span<const T> const_span(ptr, count);

    // data()
    EXPECT_NE(span.data(), nullptr);
    EXPECT_EQ(span.data(), ptr);
    EXPECT_NE(const_span.data(), nullptr);
    EXPECT_EQ(const_span.data(), ptr);

    // operator[]
    span[0].value() = T(1);
    EXPECT_FLOAT_EQ(span[0].value(), T(1));
    EXPECT_FLOAT_EQ(const_span[0].value(), T(1));
    span[count - 1].value() = T(2);
    EXPECT_FLOAT_EQ(span[count - 1].value(), T(2));
    EXPECT_FLOAT_EQ(const_span[count - 1].value(), T(2));

    // at()
    span[0].value() = T(3);
    EXPECT_FLOAT_EQ(span.at(0).value(), T(3));
    EXPECT_FLOAT_EQ(const_span.at(0).value(), T(3));
    span.at(count - 1).value() = T(4);
    EXPECT_FLOAT_EQ(span.at(count - 1).value(), T(4));
    EXPECT_FLOAT_EQ(const_span.at(count - 1).value(), T(4));

    // size()
    EXPECT_EQ(span.size(), count);
    EXPECT_EQ(const_span.size(), count);

    // size_byte()
    EXPECT_EQ(span.size_bytes(), count * sizeof(T));
    EXPECT_EQ(const_span.size_bytes(), count * sizeof(T));

    // operator==
    EXPECT_EQ(const_span, span);

    delete[] ptr;
  }
}

TEST(TestSpan, Subspan) {
  std::vector<char> vector(26);
  std::iota(vector.begin(), vector.end(), 'a');
  const Span<const char> span(vector);

  EXPECT_EQ(span.subspan(0, 3).value(), Span<const char>("abc", 3));
  EXPECT_EQ(span.subspan(23).value(),   Span<const char>("xyz", 3));
  EXPECT_EQ(span.subspan(3, 3).value(), Span<const char>("def", 3));
  EXPECT_NE(span.subspan(0, 4).value(), Span<const char>("abc", 3));
  EXPECT_NE(span.subspan(0, 3).value(), Span<const char>("xyz", 3));

  EXPECT_EQ(span.first(3).value(), Span<const char>("abc", 3));
  EXPECT_EQ(span.last(3).value(),  Span<const char>("xyz", 3));

  EXPECT_EQ(span.subspan(0, 26).value(), span);
  EXPECT_EQ(span.first(26).value(),      span);
  EXPECT_EQ(span.last(26).value(),       span);

  EXPECT_EQ(span.subspan(26).value(),    Span<const char>());
  EXPECT_EQ(span.subspan(0, 0).value(),  Span<const char>());
  EXPECT_EQ(span.subspan(13, 0).value(), Span<const char>());
  EXPECT_EQ(span.first(0).value(),       Span<const char>());
  EXPECT_EQ(span.last(0).value(),        Span<const char>());

  EXPECT_FALSE(span.subspan(27));
  EXPECT_FALSE(span.first(27));
  EXPECT_FALSE(span.last(27));
}

TEST(TestSpan, Iterator) {
  std::vector<char> vector(26);
  std::iota(vector.begin(), vector.end(), 'a');
  Span<char> span(vector);
  const Span<const char> const_span(vector);

  ASSERT_EQ(span, Span<char>(vector));

  for (auto element : span) {
    element.value() -= 32;
  }

  ASSERT_EQ(span, Span<char>(vector));

  char letter = 'A';
  for (auto element : const_span) {
    EXPECT_EQ(element.value(), letter++);
  }

  ASSERT_EQ(letter, 'Z' + 1);

  for (auto iter = const_span.rbegin(); iter != const_span.rend(); iter++) {
    auto element = *iter;
    EXPECT_EQ(element.value(), --letter);
  }

  ASSERT_EQ(letter, 'A');

  auto iter = span.begin();
  for (size_t i = 0; i < span.size(); i++) {
    auto element = iter[i];
    ASSERT_TRUE(element);
    element.value() += 32;
  }

  ASSERT_EQ(span, Span<char>(vector));

  letter = 'z' + 1;
  auto citer = const_span.rbegin();
  for (size_t i = 0; i < const_span.size(); i++) {
    auto element = citer[i];
    ASSERT_TRUE(element);
    ASSERT_EQ(element.value(), --letter);
  }

  ASSERT_EQ(letter, 'a');

  auto begin = span.begin();
  auto end = span.end();

  ASSERT_TRUE(*begin);
  ASSERT_FALSE(*end);

  span = Span<char>();

  ASSERT_FALSE(*begin);
  ASSERT_FALSE(*end);
}

TEST(TestSpan, Equality) {
  std::vector<int> vector1{1, 2, 3};
  std::vector<int> vector2{1, 2, 3};
  std::vector<int> vector3{1, 2, 3, 4};
  Span<int> span1(vector1);
  Span<const int> span2(vector2);
  Span<int> span3(vector3);

  EXPECT_TRUE(span1 == span2);
  EXPECT_TRUE(span2 == span1);
  EXPECT_TRUE(span1 != span3);
  EXPECT_TRUE(span3 != span1);
  EXPECT_TRUE(span2 != span3);
  EXPECT_TRUE(span3 != span2);
}

}  // namespace test
}  // namespace nvidia
