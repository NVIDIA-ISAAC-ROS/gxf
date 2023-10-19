/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gtest/gtest.h"

#include <vector>

#include "common/iterator.hpp"

namespace nvidia {
namespace test {

TEST(TestIterator, RandomAccessIterator) {
  struct Vector : public std::vector<int> {
    using Iterator = RandomAccessIterator<std::vector<int>>;
    using ConstIterator = ConstRandomAccessIterator<std::vector<int>>;
    using std::vector<int>::vector;
    const std::vector<int> base() const { return static_cast<std::vector<int>>(*this); }
    auto begin()         { return Iterator(*this, 0); }
    auto end()           { return Iterator(*this, this->size()); }
    auto rbegin()        { return ReverseIterator<Iterator>(end()); }
    auto rend()          { return ReverseIterator<Iterator>(begin()); }
    auto cbegin()  const { return ConstIterator(*this, 0); }
    auto cend()    const { return ConstIterator(*this, this->size()); }
    auto crbegin() const { return ReverseIterator<ConstIterator>(cend()); }
    auto crend()   const { return ReverseIterator<ConstIterator>(cbegin()); }
  };

  auto null = Vector::Iterator();

  EXPECT_FALSE(*null);
  EXPECT_FALSE(null[0]);
  EXPECT_EQ(null + 1, null);
  EXPECT_EQ(null - 1, null);

  Vector vector(1024);

  EXPECT_EQ(vector.begin() + vector.size(), vector.end());
  EXPECT_EQ(vector.size() + vector.begin(), vector.end());
  EXPECT_EQ(vector.end() - vector.size(), vector.begin());
  EXPECT_EQ(vector.end() - vector.begin(), vector.size());
  EXPECT_EQ(vector.begin() - vector.end(), -vector.size());

  EXPECT_EQ(vector.rbegin() + vector.size(), vector.rend());
  EXPECT_EQ(vector.size() + vector.rbegin(), vector.rend());
  EXPECT_EQ(vector.rend() - vector.size(), vector.rbegin());
  EXPECT_EQ(vector.rend() - vector.rbegin(), vector.size());
  EXPECT_EQ(vector.rbegin() - vector.rend(), -vector.size());

  // Sequential write forward
  size_t count = 0;
  for (auto element : vector) {
    ASSERT_TRUE(element);
    element.value() = count++;
  }
  ASSERT_EQ(count, vector.size());

  // Sequential read reverse
  for (auto criter = vector.crbegin(); criter != vector.crend(); criter++) {
    auto element = *criter;
    ASSERT_TRUE(element);
    EXPECT_EQ(element.value(), --count);
  }
  ASSERT_EQ(count, 0);

  // Sanity check
  size_t expected = 0;
  for (auto element : vector.base()) {
    EXPECT_EQ(element, expected++);
  }
  ASSERT_EQ(expected, vector.size());

  // Sequential write reverse
  for (auto riter = vector.rbegin(); riter != vector.rend(); riter++) {
    auto element = *riter;
    ASSERT_TRUE(element);
    element.value() = count++;
  }
  ASSERT_EQ(count, vector.size());

  // Sequential read forward
  for (auto citer = vector.cbegin(); citer != vector.cend(); citer++) {
    auto element = *citer;
    ASSERT_TRUE(element);
    EXPECT_EQ(element.value(), --count);
  }
  ASSERT_EQ(count, 0);

  // Sanity check
  for (auto element : vector.base()) {
    EXPECT_EQ(element, --expected);
  }
  ASSERT_EQ(expected, 0);

  // Random-access write forward
  auto iter = vector.begin();
  for (size_t i = 0; i < vector.size(); i++) {
    auto element = iter[i];
    ASSERT_TRUE(element);
    element.value() = count++;
  }
  ASSERT_EQ(count, vector.size());

  // Random-access read reverse
  auto criter = vector.crbegin();
  for (size_t i = 0; i < vector.size(); i++) {
    auto element = criter[i];
    ASSERT_TRUE(element);
    EXPECT_EQ(element.value(), --count);
  }
  ASSERT_EQ(count, 0);

  // Sanity check
  for (auto element : vector.base()) {
    EXPECT_EQ(element, expected++);
  }
  ASSERT_EQ(expected, vector.size());

  // Random-access write reverse
  auto riter = vector.rbegin();
  for (size_t i = 0; i < vector.size(); i++) {
    auto element = riter[i];
    ASSERT_TRUE(element);
    element.value() = count++;
  }
  ASSERT_EQ(count, vector.size());

  // Random-access read forward
  auto citer = vector.cbegin();
  for (size_t i = 0; i < vector.size(); i++) {
    auto element = citer[i];
    ASSERT_TRUE(element);
    EXPECT_EQ(element.value(), --count);
  }
  ASSERT_EQ(count, 0);

  // Sanity check
  for (auto element : vector.base()) {
    EXPECT_EQ(element, --expected);
  }
  ASSERT_EQ(expected, 0);

  const auto begin = vector.begin();
  const auto end = vector.end();

  ASSERT_TRUE(*begin);
  ASSERT_FALSE(*end);

  vector.clear();

  ASSERT_FALSE(*begin);
  ASSERT_FALSE(*end);
}

}  // namespace test
}  // namespace nvidia
