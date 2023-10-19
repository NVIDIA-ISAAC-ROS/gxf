/*
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gtest/gtest.h"

#include <utility>

#include "common/fixed_map.hpp"
#include "common/fixed_string.hpp"

namespace nvidia {
namespace test {

namespace {

constexpr size_t kMapSizeNone = 0;
constexpr size_t kMapSizeTiny = 1;
constexpr size_t kMapSizeSmall = 8;
constexpr size_t kMapSizeMedium = 64;
constexpr size_t kMapSizeLarge = 512;
constexpr size_t kMapSizeMassive = 4096;

}  // namespace

TEST(TestFixedMap, DefaultConstruct) {
  FixedMap<int, int> map;

  EXPECT_EQ(map.size(), 0);
  EXPECT_EQ(map.capacity(), 0);
  EXPECT_DOUBLE_EQ(map.load_factor(), 0.0);
  EXPECT_FALSE(map.at(0));
  EXPECT_FALSE(map.insert(std::make_pair(0, 0)));
  EXPECT_FALSE(map.contains(0));
  EXPECT_FALSE(map.erase(0));
}

TEST(TestFixedMap, Create) {
  FixedMap<size_t, size_t> map;

  ASSERT_TRUE(map.reserve(kMapSizeMedium));

  EXPECT_TRUE(map.empty());
  EXPECT_FALSE(map.full());

  EXPECT_EQ(map.size(), 0);
  EXPECT_EQ(map.capacity(), kMapSizeMedium);

  size_t i;
  for (i = 0; i < map.capacity(); i++) {
    ASSERT_TRUE(map.insert(std::make_pair(i, i)));
  }

  EXPECT_FALSE(map.empty());
  EXPECT_TRUE(map.full());

  EXPECT_EQ(map.size(), kMapSizeMedium);
  EXPECT_EQ(map.capacity(), kMapSizeMedium);

  ASSERT_FALSE(map.insert(std::make_pair(i, i)));

  EXPECT_EQ(map.size(), kMapSizeMedium);
  EXPECT_EQ(map.capacity(), kMapSizeMedium);

  for (i = 0; i < map.capacity(); i++) {
    auto element = map.at(i);
    ASSERT_TRUE(element);
    EXPECT_EQ(element.value(), i);
  }

  EXPECT_EQ(map.size(), kMapSizeMedium);
  EXPECT_EQ(map.capacity(), kMapSizeMedium);
}

TEST(TestFixedMap, MoveConstruct) {
  FixedMap<size_t, size_t> map;

  ASSERT_TRUE(map.reserve(kMapSizeMedium));
  const size_t capacity = map.capacity();

  for (size_t i = 0; i < map.capacity(); i++) {
    ASSERT_TRUE(map.insert(std::make_pair(i, i)));
  }

  FixedMap<size_t, size_t> map_moved = FixedMap<size_t, size_t>(std::move(map));

  EXPECT_EQ(map.size(), 0);
  EXPECT_EQ(map.capacity(), 0);

  EXPECT_EQ(map_moved.size(), capacity);
  EXPECT_EQ(map_moved.capacity(), capacity);

  for (size_t i = 0; i < map_moved.capacity(); i++) {
    auto element = map_moved.at(i);
    ASSERT_TRUE(element);
    EXPECT_EQ(element.value(), i);
  }
}

TEST(TestFixedMap, MoveAssignment) {
  FixedMap<size_t, size_t> map;

  ASSERT_TRUE(map.reserve(kMapSizeMedium));
  const size_t capacity = map.capacity();

  for (size_t i = 0; i < map.capacity(); i++) {
    ASSERT_TRUE(map.insert(std::make_pair(i, i)));
  }

  FixedMap<size_t, size_t> map_moved;
  map_moved = std::move(map);

  EXPECT_EQ(map.size(), 0);
  EXPECT_EQ(map.capacity(), 0);

  EXPECT_EQ(map_moved.size(), capacity);
  EXPECT_EQ(map_moved.capacity(), capacity);

  for (size_t i = 0; i < map_moved.capacity(); i++) {
    auto element = map_moved.at(i);
    ASSERT_TRUE(element);
    EXPECT_EQ(element.value(), i);
  }
}

TEST(TestFixedMap, Equality) {
  FixedMap<size_t, size_t> map1;
  FixedMap<size_t, size_t> map2;
  FixedMap<size_t, size_t> map3;

  ASSERT_TRUE(map1.reserve(kMapSizeMedium));
  for (size_t i = 0; i < map1.capacity(); i++) {
    ASSERT_TRUE(map1.insert(std::make_pair(i, i)));
  }

  ASSERT_TRUE(map2.reserve(kMapSizeMedium));
  for (size_t i = 0; i < map2.capacity(); i++) {
    ASSERT_TRUE(map2.insert(std::make_pair(i, i)));
  }

  ASSERT_TRUE(map3.reserve(kMapSizeMedium));
  for (size_t i = 0; i < map3.capacity(); i++) {
    ASSERT_TRUE(map3.insert(std::make_pair(i, 0)));
  }

  EXPECT_EQ(map2, map1);
  EXPECT_NE(map3, map1);
}

TEST(TestFixedMap, Reserve) {
  FixedMap<size_t, size_t> map;

  ASSERT_TRUE(map.reserve(kMapSizeMedium));

  EXPECT_EQ(map.size(), 0);
  EXPECT_EQ(map.capacity(), kMapSizeMedium);

  for (size_t i = 0; i < kMapSizeMedium; i++) {
    ASSERT_TRUE(map.insert(std::make_pair(i, i)));
  }

  EXPECT_EQ(map.size(), kMapSizeMedium);
  EXPECT_EQ(map.capacity(), kMapSizeMedium);

  ASSERT_TRUE(map.reserve(map.capacity() / 2));

  EXPECT_EQ(map.size(), kMapSizeMedium);
  EXPECT_EQ(map.capacity(), kMapSizeMedium);

  ASSERT_TRUE(map.reserve(map.capacity() * 2));

  for (size_t i = 0; i < kMapSizeMedium; i++) {
    auto element = map.at(i);
    ASSERT_TRUE(element);
    EXPECT_EQ(element.value(), i);
  }

  EXPECT_EQ(map.size(), kMapSizeMedium);
  EXPECT_EQ(map.capacity(), kMapSizeMedium * 2);
}

TEST(TestFixedMap, PointerType) {
  FixedMap<size_t, int*> map;

  ASSERT_TRUE(map.reserve(kMapSizeTiny));

  int integer = 10;
  ASSERT_TRUE(map.insert(std::make_pair(0, &integer)));

  auto element = map.at(0);
  ASSERT_TRUE(element);

  EXPECT_EQ(element.value(), &integer);
  EXPECT_EQ(*element.value(), integer);

  integer++;

  EXPECT_EQ(element.value(), &integer);
  EXPECT_EQ(*element.value(), integer);
}

TEST(TestFixedMap, Contains) {
  FixedMap<size_t, size_t> map;

  ASSERT_TRUE(map.reserve(kMapSizeLarge));

  for (size_t i = 0; i < map.capacity(); i++) {
    ASSERT_TRUE(map.insert(std::make_pair(i, i)));
  }

  for (size_t i = 0; i < map.capacity(); i++) {
    EXPECT_TRUE(map.contains(i));
  }

  for (size_t i = 0; i < map.capacity(); i++) {
    ASSERT_TRUE(map.erase(i));
  }

  for (size_t i = 0; i < map.capacity(); i++) {
    EXPECT_FALSE(map.contains(i));
  }
}

TEST(TestFixedMap, Update) {
  FixedMap<size_t, char> map;
  ASSERT_TRUE(map.reserve(kMapSizeSmall));

  ASSERT_TRUE(map.emplace(std::make_pair(0, 'A')));
  ASSERT_TRUE(map.emplace(std::make_pair(1, 'B')));
  ASSERT_EQ(map.size(), 2);

  EXPECT_FALSE(map.update(2, 'D'));

  ASSERT_TRUE(map.update(0, 'C'));
  ASSERT_EQ(map.size(), 2);

  auto C = map.at(0);
  ASSERT_TRUE(C);
  EXPECT_EQ(C.value(), 'C');

  auto B = map.at(1);
  ASSERT_TRUE(B);
  EXPECT_EQ(B.value(), 'B');
}

TEST(TestFixedMap, InsertOrAssign) {
  FixedMap<size_t, char> map;
  ASSERT_TRUE(map.reserve(kMapSizeSmall));
  ASSERT_TRUE(map.emplace(std::make_pair(0, 'A')));
  ASSERT_TRUE(map.emplace(std::make_pair(1, 'B')));
  ASSERT_EQ(map.size(), 2);

  auto A = map.at(0);
  ASSERT_TRUE(A);
  EXPECT_EQ(A.value(), 'A');

  EXPECT_TRUE(map.insert_or_assign(0, 'C'));
  auto C = map.at(0);
  ASSERT_TRUE(C);
  EXPECT_EQ(C.value(), 'C');

  EXPECT_TRUE(map.insert_or_assign(3, 'D'));
  auto D = map.at(3);
  ASSERT_TRUE(D);
  EXPECT_EQ(D.value(), 'D');
}

TEST(TestFixedMap, Emplace) {
  FixedMap<size_t, char> map;

  ASSERT_TRUE(map.reserve(kMapSizeSmall));

  ASSERT_TRUE(map.emplace(std::make_pair(0, 'A')));
  ASSERT_TRUE(map.emplace(std::make_pair(1, 'B')));
  ASSERT_EQ(map.size(), 2);

  ASSERT_FALSE(map.emplace(std::make_pair(1, 'C')));
  ASSERT_EQ(map.size(), 2);

  ASSERT_TRUE(map.emplace(std::make_pair(2, 'C')));
  ASSERT_EQ(map.size(), 3);

  ASSERT_TRUE(map.emplace(std::make_pair(3, 'D')));
  ASSERT_TRUE(map.emplace(std::make_pair(7, 'E')));
  ASSERT_TRUE(map.emplace(std::make_pair(5, 'F')));
  ASSERT_EQ(map.size(), 6);

  auto A = map.at(0);
  ASSERT_TRUE(A);
  EXPECT_EQ(A.value(), 'A');

  auto B = map.at(1);
  ASSERT_TRUE(B);
  EXPECT_EQ(B.value(), 'B');

  auto C = map.at(2);
  ASSERT_TRUE(C);
  EXPECT_EQ(C.value(), 'C');

  auto D = map.at(3);
  ASSERT_TRUE(D);
  EXPECT_EQ(D.value(), 'D');

  auto E = map.at(7);
  ASSERT_TRUE(E);
  EXPECT_EQ(E.value(), 'E');

  auto F = map.at(5);
  ASSERT_TRUE(F);
  EXPECT_EQ(F.value(), 'F');

  EXPECT_FALSE(map.at(4));
  EXPECT_FALSE(map.at(6));
  EXPECT_FALSE(map.at(8));
}

TEST(TestFixedMap, Erase) {
  FixedMap<size_t, char> map;

  ASSERT_TRUE(map.reserve(kMapSizeSmall));

  ASSERT_TRUE(map.insert(std::make_pair(0, 'A')));
  ASSERT_TRUE(map.insert(std::make_pair(1, 'B')));
  ASSERT_TRUE(map.insert(std::make_pair(2, 'C')));
  ASSERT_TRUE(map.insert(std::make_pair(3, 'D')));
  ASSERT_TRUE(map.insert(std::make_pair(4, 'E')));
  ASSERT_TRUE(map.insert(std::make_pair(5, 'F')));
  ASSERT_EQ(map.size(), 6);

  ASSERT_FALSE(map.erase(6));
  ASSERT_EQ(map.size(), 6);

  ASSERT_TRUE(map.erase(5));
  ASSERT_EQ(map.size(), 5);

  ASSERT_TRUE(map.erase(0));
  ASSERT_TRUE(map.erase(1));
  ASSERT_TRUE(map.erase(2));
  ASSERT_EQ(map.size(), 2);

  auto D = map.at(3);
  ASSERT_TRUE(D);
  EXPECT_EQ(D.value(), 'D');

  auto E = map.at(4);
  ASSERT_TRUE(E);
  EXPECT_EQ(E.value(), 'E');

  EXPECT_FALSE(map.at(0));
  EXPECT_FALSE(map.at(1));
  EXPECT_FALSE(map.at(2));
  EXPECT_FALSE(map.at(5));
}

TEST(TestFixedMap, Insert) {
  FixedMap<size_t, char> map;

  ASSERT_TRUE(map.reserve(kMapSizeSmall));

  ASSERT_TRUE(map.insert(std::make_pair(0, 'A')));
  ASSERT_TRUE(map.insert(std::make_pair(1, 'B')));
  ASSERT_TRUE(map.insert(std::make_pair(2, 'C')));
  ASSERT_TRUE(map.insert(std::make_pair(3, 'D')));
  ASSERT_TRUE(map.insert(std::make_pair(4, 'E')));
  ASSERT_TRUE(map.insert(std::make_pair(5, 'F')));
  ASSERT_TRUE(map.insert(std::make_pair(6, 'G')));

  ASSERT_EQ(map.size(), map.capacity() - 1);

  ASSERT_FALSE(map.insert(std::make_pair(0, 'G')));
  ASSERT_TRUE(map.insert(std::make_pair(8, 'G')));

  auto A = map.at(0);
  ASSERT_TRUE(A);
  EXPECT_EQ(A.value(), 'A');

  auto G = map.at(8);
  ASSERT_TRUE(G);
  EXPECT_EQ(G.value(), 'G');

  ASSERT_TRUE(map.erase(8));

  ASSERT_TRUE(map.insert(std::make_pair(7, 'D')));

  auto D3 = map.at(3);
  ASSERT_TRUE(D3);
  EXPECT_EQ(D3.value(), 'D');

  auto D7 = map.at(7);
  ASSERT_TRUE(D7);
  EXPECT_EQ(D7.value(), 'D');
}

TEST(TestFixedMap, Const) {
  FixedMap<size_t, size_t> map;

  ASSERT_TRUE(map.reserve(kMapSizeLarge));

  const auto& const_map = map;

  for (size_t i = 0; i < map.capacity(); i++) {
    ASSERT_TRUE(map.insert(std::make_pair(i, i)));
  }

  EXPECT_EQ(const_map.size(), map.capacity());
  EXPECT_EQ(const_map.capacity(), map.capacity());

  auto front = const_map.at(0);
  ASSERT_TRUE(front);
  EXPECT_EQ(front.value(), 0);

  auto back = const_map.at(map.capacity() - 1);
  ASSERT_TRUE(back);
  EXPECT_EQ(back.value(), map.capacity() - 1);

  for (size_t i = 0; i < const_map.capacity(); i++) {
    auto element = const_map.at(i);
    ASSERT_TRUE(element);
    EXPECT_EQ(element.value(), i);
  }
}

TEST(TestFixedMap, CopyFrom) {
  FixedMap<size_t, size_t> map1;
  FixedMap<size_t, size_t> map2;

  ASSERT_TRUE(map1.reserve(kMapSizeMedium));

  for (size_t i = 0; i < map1.capacity(); i++) {
    ASSERT_TRUE(map1.insert(std::make_pair(i, i)));
  }

  EXPECT_NE(map2, map1);

  ASSERT_FALSE(map2.copy_from(map1));

  ASSERT_TRUE(map2.reserve(kMapSizeMedium));

  ASSERT_TRUE(map2.copy_from(map1));

  EXPECT_EQ(map2, map1);
}

TEST(TestFixedMap, SubscriptOperator) {
  FixedMap<size_t, size_t> map;

  ASSERT_TRUE(map.reserve(kMapSizeSmall));

  for (size_t i = 0; i < map.capacity(); i++) {
    map[i] = i;
    ASSERT_TRUE(map[i]);
  }

  ASSERT_TRUE(map.full());

  size_t& ref = map[0].value();
  EXPECT_EQ(ref, 0);

  ref = 10;
  EXPECT_EQ(ref, 10);

  ASSERT_TRUE(map.at(0));
  EXPECT_EQ(ref, map.at(0).value());
}

TEST(TestFixedMap, Clear) {
  FixedMap<size_t, size_t> map;

  ASSERT_TRUE(map.reserve(kMapSizeMassive));

  for (size_t i = 0; i < map.capacity(); i++) {
    ASSERT_TRUE(map.insert(std::make_pair(i, i)));
  }

  EXPECT_EQ(map.capacity(), kMapSizeMassive);
  EXPECT_TRUE(map.full());

  map.clear();

  EXPECT_EQ(map.capacity(), kMapSizeMassive);
  EXPECT_TRUE(map.empty());
}

TEST(TestFixedMap, Hash) {
  constexpr size_t kStringSize = 8;
  FixedMap<FixedString<kStringSize>, int, FixedString<kStringSize>::Hash> map;

  ASSERT_TRUE(map.reserve(kMapSizeMedium));

  ASSERT_TRUE(map.emplace("the", 0));
  ASSERT_TRUE(map.emplace("quick", 1));
  ASSERT_TRUE(map.emplace("brown", 2));
  ASSERT_TRUE(map.emplace("fox", 3));
  ASSERT_TRUE(map.emplace("jumps", 4));
  ASSERT_TRUE(map.emplace("over", 5));
  ASSERT_TRUE(map.emplace("a", 6));
  ASSERT_TRUE(map.emplace("lazy", 7));
  ASSERT_TRUE(map.emplace("dog", 8));

  EXPECT_EQ(map["the"], 0);
  EXPECT_EQ(map["quick"], 1);
  EXPECT_EQ(map["brown"], 2);
  EXPECT_EQ(map["fox"], 3);
  EXPECT_EQ(map["jumps"], 4);
  EXPECT_EQ(map["over"], 5);
  EXPECT_EQ(map["a"], 6);
  EXPECT_EQ(map["lazy"], 7);
  EXPECT_EQ(map["dog"], 8);
}

TEST(TestFixedMap, Iterator) {
  FixedMap<size_t, size_t> map;

  ASSERT_TRUE(map.reserve(kMapSizeLarge));

  for (size_t i = 0; i < map.capacity(); i += 2) {
    ASSERT_TRUE(map.insert(std::make_pair(i, 0)));
  }

  ASSERT_EQ(map.size(), map.capacity() / 2);

  int count = 0;
  for (auto element : map) {
    ASSERT_TRUE(element);
    element->second = count++;
  }

  ASSERT_EQ(count, map.size());

  for (auto criter = map.crbegin(); criter != map.crend(); criter++) {
    auto element = *criter;
    ASSERT_TRUE(element);
    ASSERT_EQ(element->second, --count);
  }

  ASSERT_EQ(count, 0);

  for (auto riter = map.rbegin(); riter != map.rend(); riter++) {
    auto element = *riter;
    ASSERT_TRUE(element);
    element->second = count++;
  }

  ASSERT_EQ(count, map.size());

  for (auto citer = map.cbegin(); citer != map.cend(); citer++) {
    auto element = *citer;
    ASSERT_TRUE(element);
    ASSERT_EQ(element->second, --count);
  }

  ASSERT_EQ(count, 0);

  auto begin = map.begin();
  auto end = map.end();

  ASSERT_TRUE(*begin);
  ASSERT_FALSE(*end);

  map.clear();

  ASSERT_FALSE(*begin);
  ASSERT_FALSE(*end);
}

}  // namespace test
}  // namespace nvidia
