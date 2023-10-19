/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gtest/gtest.h"

#include <cstring>
#include <string>
#include <utility>

#include "common/fixed_vector.hpp"

namespace nvidia {
namespace test {

namespace {

template <ssize_t N>
void TestCopyConstruct(FixedVector<int, N>& vector) {
  const size_t kCapacity = vector.capacity();

  for (size_t i = 0; i < vector.capacity(); i++) {
    ASSERT_TRUE(vector.push_back(static_cast<int>(i)));
  }

  auto vector_copy = FixedVector<int, N>(vector);

  ASSERT_EQ(vector.size(), kCapacity);
  ASSERT_EQ(vector.capacity(), kCapacity);

  for (size_t i = 0; i < vector.size(); i++) {
    auto element = vector.at(i);
    ASSERT_TRUE(element);
    ASSERT_EQ(element.value(), static_cast<int>(i));
  }

  ASSERT_EQ(vector_copy.size(), kCapacity);
  ASSERT_EQ(vector_copy.capacity(), kCapacity);

  for (size_t i = 0; i < vector_copy.size(); i++) {
    auto element = vector_copy.at(i);
    ASSERT_TRUE(element);
    ASSERT_EQ(element.value(), static_cast<int>(i));
  }
}

template <ssize_t N>
void TestMoveConstruct(FixedVector<int, N>& vector) {
  const size_t kCapacity = vector.capacity();

  for (size_t i = 0; i < vector.capacity(); i++) {
    ASSERT_TRUE(vector.push_back(static_cast<int>(i)));
  }

  auto vector_move = FixedVector<int, N>(std::move(vector));

  ASSERT_EQ(vector.size(), 0);
  ASSERT_EQ(vector.capacity(), N == kFixedVectorHeap ? 0 : kCapacity);

  ASSERT_EQ(vector_move.size(), kCapacity);
  ASSERT_EQ(vector_move.capacity(), kCapacity);

  for (size_t i = 0; i < vector_move.size(); i++) {
    auto element = vector_move.at(i);
    ASSERT_TRUE(element);
    ASSERT_EQ(element.value(), static_cast<int>(i));
  }
}

template <ssize_t N>
void TestCopyAssignment(FixedVector<int, N>& vector) {
  const size_t kCapacity = vector.capacity();

  for (size_t i = 0; i < vector.capacity(); i++) {
    ASSERT_TRUE(vector.push_back(static_cast<int>(i)));
  }

  FixedVector<int, N> vector_copy;
  vector_copy = vector;

  ASSERT_EQ(vector.size(), kCapacity);
  ASSERT_EQ(vector.capacity(), kCapacity);

  for (size_t i = 0; i < vector.size(); i++) {
    auto element = vector.at(i);
    ASSERT_TRUE(element);
    ASSERT_EQ(element.value(), static_cast<int>(i));
  }

  ASSERT_EQ(vector_copy.size(), kCapacity);
  ASSERT_EQ(vector_copy.capacity(), kCapacity);

  for (size_t i = 0; i < vector_copy.size(); i++) {
    auto element = vector_copy.at(i);
    ASSERT_TRUE(element);
    ASSERT_EQ(element.value(), static_cast<int>(i));
  }
}

template <ssize_t N>
void TestMoveAssignment(FixedVector<int, N>& vector) {
  const size_t kCapacity = vector.capacity();

  for (size_t i = 0; i < vector.capacity(); i++) {
    ASSERT_TRUE(vector.push_back(static_cast<int>(i)));
  }

  FixedVector<int, N> vector_move;
  vector_move = std::move(vector);

  ASSERT_EQ(vector.size(), 0);
  ASSERT_EQ(vector.capacity(), N == kFixedVectorHeap ? 0 : kCapacity);

  ASSERT_EQ(vector_move.size(), kCapacity);
  ASSERT_EQ(vector_move.capacity(), kCapacity);

  for (size_t i = 0; i < vector_move.size(); i++) {
    auto element = vector_move.at(i);
    ASSERT_TRUE(element);
    ASSERT_EQ(element.value(), static_cast<int>(i));
  }
}

template <ssize_t N>
void TestPointerType(FixedVector<int*, N>& vector) {
  int integer = 10;
  ASSERT_TRUE(vector.push_back(&integer));

  auto back = vector.back();
  ASSERT_TRUE(back);
  ASSERT_EQ(back.value(), &integer);
  ASSERT_EQ(*back.value(), integer);

  integer++;
  ASSERT_EQ(back.value(), &integer);
  ASSERT_EQ(*back.value(), integer);
}

template <ssize_t N>
void TestContiguousMemory(FixedVector<uint8_t, N>& vector) {
  uint8_t* data = new uint8_t[vector.capacity()];
  std::memset(data, 0xAA, vector.capacity());

  for (size_t i = 0; i < vector.capacity(); i++) {
    ASSERT_TRUE(vector.push_back(0x00));
  }

  ASSERT_EQ(vector.size(), vector.capacity());
  ASSERT_NE(std::memcmp(vector.data(), data, vector.capacity()), 0);

  for (size_t i = 0; i < vector.capacity(); i++) {
    auto element = vector.at(i);
    ASSERT_TRUE(element);
    element.value() = data[i];
  }

  ASSERT_EQ(vector.size(), vector.capacity());
  ASSERT_EQ(std::memcmp(vector.data(), data, vector.capacity()), 0);

  delete data;
}

template <ssize_t N>
void TestEmplace(FixedVector<std::string, N>& vector) {
  ASSERT_TRUE(vector.capacity() >= 6);

  ASSERT_TRUE(vector.emplace(0, "A"));
  ASSERT_TRUE(vector.emplace(1, "B"));
  ASSERT_EQ(vector.size(), 2);

  ASSERT_FALSE(vector.emplace(3, "C"));
  ASSERT_EQ(vector.size(), 2);

  ASSERT_TRUE(vector.emplace(1, "C"));
  ASSERT_EQ(vector.size(), 3);

  ASSERT_TRUE(vector.emplace(0, "D"));
  ASSERT_TRUE(vector.emplace(0, "E"));
  ASSERT_TRUE(vector.emplace(0, "F"));
  ASSERT_EQ(vector.size(), 6);

  auto front = vector.front();
  ASSERT_TRUE(front);
  ASSERT_EQ(front.value(), "F");

  auto back = vector.back();
  ASSERT_TRUE(back);
  ASSERT_EQ(back.value(), "B");
}

template <ssize_t N>
void TestErase(FixedVector<std::string, N>& vector) {
  ASSERT_TRUE(vector.capacity() >= 6);

  ASSERT_TRUE(vector.push_back("A"));
  ASSERT_TRUE(vector.push_back("B"));
  ASSERT_TRUE(vector.push_back("C"));
  ASSERT_TRUE(vector.push_back("D"));
  ASSERT_TRUE(vector.push_back("E"));
  ASSERT_TRUE(vector.push_back("F"));
  ASSERT_EQ(vector.size(), 6);

  ASSERT_FALSE(vector.erase(6));
  ASSERT_EQ(vector.size(), 6);

  ASSERT_TRUE(vector.erase(5));
  ASSERT_EQ(vector.size(), 5);

  ASSERT_TRUE(vector.erase(0));
  ASSERT_TRUE(vector.erase(0));
  ASSERT_TRUE(vector.erase(0));
  ASSERT_EQ(vector.size(), 2);

  auto front = vector.front();
  ASSERT_TRUE(front);
  ASSERT_EQ(front.value(), "D");

  auto back = vector.back();
  ASSERT_TRUE(back);
  ASSERT_EQ(back.value(), "E");
}

template <ssize_t N>
void TestInsert(FixedVector<std::string, N>& vector) {
  ASSERT_TRUE(vector.capacity() >= 6);

  ASSERT_TRUE(vector.push_back("A"));
  ASSERT_TRUE(vector.push_back("B"));
  ASSERT_TRUE(vector.push_back("C"));
  ASSERT_TRUE(vector.push_back("D"));
  ASSERT_TRUE(vector.push_back("E"));
  ASSERT_TRUE(vector.push_back("F"));

  ASSERT_TRUE(vector.insert(1, "X"));
  auto X = vector.at(1);
  ASSERT_TRUE(X);
  ASSERT_EQ(X.value(), "X");

  auto B = vector.at(2);
  ASSERT_TRUE(B);
  ASSERT_EQ(B.value(), "B");
}

template <ssize_t N>
void TestConst(FixedVector<int, N>& vector) {
  const auto& const_vector = vector;

  for (size_t i = 0; i < vector.capacity(); i++) {
    ASSERT_TRUE(vector.push_back(static_cast<int>(i)));
  }

  ASSERT_EQ(const_vector.size(), vector.capacity());
  ASSERT_EQ(const_vector.capacity(), vector.capacity());

  auto front = const_vector.front();
  ASSERT_TRUE(front);
  ASSERT_EQ(front.value(), 0);

  auto back = const_vector.back();
  ASSERT_TRUE(back);
  ASSERT_EQ(back.value(), vector.capacity() - 1);

  for (size_t i = 0; i < const_vector.size(); i++) {
    auto element = const_vector.at(i);
    ASSERT_TRUE(element);
    ASSERT_EQ(element.value(), static_cast<int>(i));
  }
}

template <ssize_t N>
void TestSubscriptOperator(FixedVector<int, N>& vector) {
  const auto& const_vector = vector;

  for (size_t i = 0; i < vector.capacity(); i++) {
    ASSERT_TRUE(vector.push_back(i));
  }

  ASSERT_TRUE(vector[0]);

  int& ref = vector[0].value();
  const int& const_ref = const_vector[0].value();

  ASSERT_EQ(ref, 0);
  ASSERT_EQ(const_ref, 0);

  ref = 10;

  ASSERT_EQ(ref, 10);
  ASSERT_EQ(const_ref, 10);

  ASSERT_TRUE(vector.at(0));
  ASSERT_EQ(ref, vector.at(0).value());
  ASSERT_EQ(const_ref, vector.at(0).value());
}

template <ssize_t N>
void TestIterator(FixedVector<int, N>& vector) {
  for (size_t i = 0; i < vector.capacity(); i++) {
    ASSERT_TRUE(vector.push_back(0));
  }

  int count = 0;
  for (auto element : vector) {
    ASSERT_TRUE(element);
    element.value() = count++;
  }

  ASSERT_EQ(count, vector.capacity());

  for (auto criter = vector.crbegin(); criter != vector.crend(); criter++) {
    auto element = *criter;
    ASSERT_TRUE(element);
    ASSERT_EQ(element.value(), --count);
  }

  ASSERT_EQ(count, 0);

  for (auto riter = vector.rbegin(); riter != vector.rend(); riter++) {
    auto element = *riter;
    ASSERT_TRUE(element);
    element.value() = count++;
  }

  ASSERT_EQ(count, vector.capacity());

  for (auto citer = vector.cbegin(); citer != vector.cend(); citer++) {
    auto element = *citer;
    ASSERT_TRUE(element);
    ASSERT_EQ(element.value(), --count);
  }

  ASSERT_EQ(count, 0);

  auto begin = vector.begin();
  auto end = vector.end();

  ASSERT_TRUE(*begin);
  ASSERT_FALSE(*end);

  vector.clear();

  ASSERT_FALSE(*begin);
  ASSERT_FALSE(*end);
}

}  // namespace

TEST(TestFixedVector, StackCreate) {
  constexpr size_t kVectorSize = 1024;
  FixedVector<int, kVectorSize> vector;

  ASSERT_TRUE(vector.empty());
  ASSERT_FALSE(vector.full());

  ASSERT_EQ(vector.size(), 0);
  ASSERT_EQ(vector.capacity(), kVectorSize);

  for (size_t i = 0; i < vector.capacity(); i++) {
    ASSERT_TRUE(vector.push_back(static_cast<int>(i)));
  }

  ASSERT_FALSE(vector.empty());
  ASSERT_TRUE(vector.full());

  ASSERT_EQ(vector.size(), kVectorSize);
  ASSERT_EQ(vector.capacity(), kVectorSize);

  ASSERT_FALSE(vector.push_back(0));

  ASSERT_EQ(vector.size(), kVectorSize);
  ASSERT_EQ(vector.capacity(), kVectorSize);

  for (size_t i = 0; i < vector.capacity(); i++) {
    auto element = vector.at(i);
    ASSERT_TRUE(element);
    ASSERT_EQ(element.value(), static_cast<int>(i));
  }
}

TEST(TestFixedVector, StackDestroy) {
  static size_t construct_count = 0;
  static size_t destruct_count = 0;
  struct DefaultType {
    DefaultType() { construct_count++; }
    ~DefaultType() { destruct_count++; }
    DefaultType(const DefaultType&) = delete;
    DefaultType(DefaultType&&) = default;
    DefaultType& operator=(const DefaultType&) = delete;
    DefaultType& operator=(DefaultType&&) = delete;
  };

  constexpr size_t kVectorSize = 100;
  size_t counter;
  {
    FixedVector<DefaultType, kVectorSize> vector;

    ASSERT_EQ(construct_count, 0);
    ASSERT_EQ(destruct_count, 0);

    for (counter = 0; counter < kVectorSize; counter++) {
      ASSERT_TRUE(vector.emplace_back());
    }

    ASSERT_EQ(construct_count, counter);
    ASSERT_EQ(destruct_count, 0);
  }

  ASSERT_EQ(construct_count, counter);
  ASSERT_EQ(destruct_count, counter);
}

TEST(TestFixedVector, StackCopyConstruct) {
  FixedVector<int, 1024> vector;
  TestCopyConstruct(vector);
}

TEST(TestFixedVector, StackMoveConstruct) {
  FixedVector<int, 1024> vector;
  TestMoveConstruct(vector);
}

TEST(TestFixedVector, StackCopyAssignment) {
  FixedVector<int, 1024> vector;
  TestCopyAssignment(vector);
}

TEST(TestFixedVector, StackMoveAssignment) {
  FixedVector<int, 1024> vector;
  TestMoveAssignment(vector);
}

TEST(TestFixedVector, StackEquality) {
  constexpr size_t kVectorSize = 1024;
  FixedVector<int, kVectorSize> vector1;
  FixedVector<int, kVectorSize * 2> vector2;
  FixedVector<int, kVectorSize> vector3;

  for (size_t i = 0; i < kVectorSize; i++) {
    ASSERT_TRUE(vector1.push_back(static_cast<int>(i)));
  }

  for (size_t i = 0; i < kVectorSize; i++) {
    ASSERT_TRUE(vector2.push_back(static_cast<int>(i)));
  }

  for (size_t i = 0; i < kVectorSize; i++) {
    ASSERT_TRUE(vector3.push_back(0));
  }

  ASSERT_EQ(vector2, vector1);
  ASSERT_NE(vector3, vector1);
}

TEST(TestFixedVector, StackCopyableType) {
  static size_t construct_count = 0;
  static size_t destruct_count = 0;
  static size_t copy_count = 0;
  struct CopyableType {
    CopyableType() { construct_count++; }
    ~CopyableType() { destruct_count++; }
    CopyableType(const CopyableType&) { copy_count++; }
    CopyableType(CopyableType&&) = default;
    CopyableType& operator=(const CopyableType&) = delete;
    CopyableType& operator=(CopyableType&&) = delete;
  };

  FixedVector<CopyableType, 100> vector;

  ASSERT_EQ(construct_count, 0);
  ASSERT_EQ(destruct_count, 0);
  ASSERT_EQ(copy_count, 0);

  CopyableType object;

  ASSERT_EQ(construct_count, 1);
  ASSERT_EQ(destruct_count, 0);
  ASSERT_EQ(copy_count, 0);

  ASSERT_TRUE(vector.push_back(object));
  ASSERT_EQ(construct_count, 1);
  ASSERT_EQ(destruct_count, 0);
  ASSERT_EQ(copy_count, 1);

  ASSERT_TRUE(vector.pop_back());
  ASSERT_EQ(construct_count, 1);
  ASSERT_EQ(destruct_count, 1);
  ASSERT_EQ(copy_count, 1);
}

TEST(TestFixedVector, StackMovableType) {
  static size_t construct_count = 0;
  static size_t destruct_count = 0;
  static size_t move_count = 0;
  struct MovableType {
    MovableType() { construct_count++; }
    ~MovableType() { destruct_count++; }
    MovableType(const MovableType&) = delete;
    MovableType(MovableType&&) { move_count++; }
    MovableType& operator=(const MovableType&) = delete;
    MovableType& operator=(MovableType&&) = delete;
  };

  FixedVector<MovableType, 100> vector;

  ASSERT_EQ(construct_count, 0);
  ASSERT_EQ(destruct_count, 0);
  ASSERT_EQ(move_count, 0);

  MovableType object;

  ASSERT_EQ(construct_count, 1);
  ASSERT_EQ(destruct_count, 0);
  ASSERT_EQ(move_count, 0);

  ASSERT_TRUE(vector.push_back(std::move(object)));
  ASSERT_EQ(construct_count, 1);
  ASSERT_EQ(destruct_count, 0);
  ASSERT_EQ(move_count, 1);

  ASSERT_TRUE(vector.pop_back());
  ASSERT_EQ(construct_count, 1);
  ASSERT_EQ(destruct_count, 1);
  ASSERT_EQ(move_count, 1);
}

TEST(TestFixedVector, StackPointerType) {
  FixedVector<int*, 1> vector;
  TestPointerType(vector);
}

TEST(TestFixedVector, StackContiguousMemory) {
  FixedVector<uint8_t, 1024> vector;
  TestContiguousMemory(vector);
}

TEST(TestFixedVector, StackEmplace) {
  FixedVector<std::string, 10> vector;
  TestEmplace(vector);
}

TEST(TestFixedVector, StackErase) {
  FixedVector<std::string, 10> vector;
  TestErase(vector);
}

TEST(TestFixedVector, StackInsert) {
  FixedVector<std::string, 10> vector;
  TestInsert(vector);
}

TEST(TestFixedVector, StackResize) {
  constexpr size_t kVectorSize = 1024;
  uint8_t data[kVectorSize];
  std::memset(data, 0xAA, kVectorSize);

  FixedVector<uint8_t, kVectorSize> vector;
  ASSERT_TRUE(vector.resize(kVectorSize, 0xAA));

  ASSERT_EQ(vector.size(), kVectorSize);
  ASSERT_EQ(std::memcmp(vector.data(), data, kVectorSize), 0);

  ASSERT_TRUE(vector.resize(kVectorSize / 2));

  ASSERT_EQ(vector.size(), kVectorSize / 2);
  ASSERT_EQ(std::memcmp(vector.data(), data, kVectorSize / 2), 0);
}

TEST(TestFixedVector, StackConst) {
  FixedVector<int, 1024> vector;
  TestConst(vector);
}

TEST(TestFixedVector, StackSubscriptOperator) {
  FixedVector<int, 16> vector;
  TestSubscriptOperator(vector);
}

TEST(TestFixedVector, StackIterator) {
  FixedVector<int, 1024> vector;
  TestIterator(vector);
}

TEST(TestFixedVector, HeapDefaultConstruct) {
  FixedVector<int> vector;

  ASSERT_EQ(vector.data(), nullptr);
  ASSERT_EQ(vector.size(), 0);
  ASSERT_EQ(vector.capacity(), 0);

  ASSERT_FALSE(vector.front());
  ASSERT_FALSE(vector.back());
  ASSERT_FALSE(vector.push_back(100));
  ASSERT_FALSE(vector.pop_back());
}

TEST(TestFixedVector, HeapCreate) {
  constexpr size_t kVectorSize = 1024;
  FixedVector<int> vector;

  ASSERT_TRUE(vector.reserve(kVectorSize));

  ASSERT_TRUE(vector.empty());
  ASSERT_FALSE(vector.full());

  ASSERT_EQ(vector.size(), 0);
  ASSERT_EQ(vector.capacity(), kVectorSize);

  for (size_t i = 0; i < vector.capacity(); i++) {
    ASSERT_TRUE(vector.push_back(static_cast<int>(i)));
  }

  ASSERT_FALSE(vector.empty());
  ASSERT_TRUE(vector.full());

  ASSERT_EQ(vector.size(), kVectorSize);
  ASSERT_EQ(vector.capacity(), kVectorSize);

  ASSERT_FALSE(vector.push_back(0));

  ASSERT_EQ(vector.size(), kVectorSize);
  ASSERT_EQ(vector.capacity(), kVectorSize);

  for (size_t i = 0; i < vector.capacity(); i++) {
    auto element = vector.at(i);
    ASSERT_TRUE(element);
    ASSERT_EQ(element.value(), static_cast<int>(i));
  }
}

TEST(TestFixedVector, HeapDestroy) {
  static size_t construct_count = 0;
  static size_t destruct_count = 0;
  struct DefaultType {
    DefaultType() { construct_count++; }
    ~DefaultType() { destruct_count++; }
    DefaultType(const DefaultType&) = delete;
    DefaultType(DefaultType&&) = default;
    DefaultType& operator=(const DefaultType&) = delete;
    DefaultType& operator=(DefaultType&&) = delete;
  };

  constexpr size_t kVectorSize = 100;
  size_t counter;

  {
    FixedVector<DefaultType> vector;
    ASSERT_TRUE(vector.reserve(kVectorSize));

    ASSERT_EQ(construct_count, 0);
    ASSERT_EQ(destruct_count, 0);

    for (counter = 0; counter < kVectorSize; counter++) {
      ASSERT_TRUE(vector.emplace_back());
    }

    ASSERT_EQ(construct_count, counter);
    ASSERT_EQ(destruct_count, 0);
  }

  ASSERT_EQ(construct_count, counter);
  ASSERT_EQ(destruct_count, counter);
}

TEST(TestFixedVector, HeapMoveConstruct) {
  FixedVector<int> vector;
  ASSERT_TRUE(vector.reserve(1024));
  TestMoveConstruct(vector);
}

TEST(TestFixedVector, HeapMoveAssignment) {
  FixedVector<int> vector;
  ASSERT_TRUE(vector.reserve(1024));
  TestMoveAssignment(vector);
}

TEST(TestFixedVector, HeapEquality) {
  constexpr size_t kVectorSize = 1024;
  FixedVector<int> vector1;
  FixedVector<int> vector2;
  FixedVector<int> vector3;

  ASSERT_TRUE(vector1.reserve(kVectorSize));
  for (size_t i = 0; i < kVectorSize; i++) {
    ASSERT_TRUE(vector1.push_back(static_cast<int>(i)));
  }

  ASSERT_TRUE(vector2.reserve(kVectorSize * 2));
  for (size_t i = 0; i < kVectorSize; i++) {
    ASSERT_TRUE(vector2.push_back(static_cast<int>(i)));
  }

  ASSERT_TRUE(vector3.reserve(kVectorSize));
  for (size_t i = 0; i < kVectorSize; i++) {
    ASSERT_TRUE(vector3.push_back(0));
  }

  ASSERT_EQ(vector2, vector1);
  ASSERT_NE(vector3, vector1);
}

TEST(TestFixedVector, HeapReserve) {
  constexpr size_t kVectorSize = 100;
  FixedVector<int> vector;

  ASSERT_TRUE(vector.reserve(kVectorSize));

  ASSERT_EQ(vector.size(), 0);
  ASSERT_EQ(vector.capacity(), kVectorSize);

  for (size_t i = 0; i < kVectorSize; i++) {
    ASSERT_TRUE(vector.push_back(i));
  }

  ASSERT_EQ(vector.size(), kVectorSize);
  ASSERT_EQ(vector.capacity(), kVectorSize);

  ASSERT_TRUE(vector.reserve(vector.capacity() * 2));

  for (size_t i = 0; i < kVectorSize; i++) {
    auto element = vector.at(i);
    ASSERT_TRUE(element);
    ASSERT_EQ(element.value(), i);
  }

  ASSERT_EQ(vector.size(), kVectorSize);
  ASSERT_EQ(vector.capacity(), kVectorSize * 2);
}

TEST(TestFixedVector, HeapShrinkToFit) {
  constexpr size_t kVectorSize = 100;
  FixedVector<int> vector;

  ASSERT_TRUE(vector.reserve(kVectorSize));

  for (size_t i = 0; i < kVectorSize / 2; i++) {
    ASSERT_TRUE(vector.push_back(i));
  }

  ASSERT_EQ(vector.size(), kVectorSize / 2);
  ASSERT_EQ(vector.capacity(), kVectorSize);

  ASSERT_TRUE(vector.shrink_to_fit());

  ASSERT_EQ(vector.size(), kVectorSize / 2);
  ASSERT_EQ(vector.capacity(), kVectorSize / 2);
}

TEST(TestFixedVector, HeapCopyableType) {
  static size_t construct_count = 0;
  static size_t destruct_count = 0;
  static size_t copy_count = 0;
  struct CopyableType {
    CopyableType() { construct_count++; }
    ~CopyableType() { destruct_count++; }
    CopyableType(const CopyableType&) { copy_count++; }
    CopyableType(CopyableType&&) = default;
    CopyableType& operator=(const CopyableType&) = delete;
    CopyableType& operator=(CopyableType&&) = delete;
  };

  constexpr size_t kVectorSize = 1;
  FixedVector<CopyableType> vector;

  ASSERT_TRUE(vector.reserve(kVectorSize));

  ASSERT_EQ(construct_count, 0);
  ASSERT_EQ(destruct_count, 0);
  ASSERT_EQ(copy_count, 0);

  CopyableType object;

  ASSERT_EQ(construct_count, 1);
  ASSERT_EQ(destruct_count, 0);
  ASSERT_EQ(copy_count, 0);

  ASSERT_TRUE(vector.push_back(object));
  ASSERT_EQ(construct_count, 1);
  ASSERT_EQ(destruct_count, 0);
  ASSERT_EQ(copy_count, 1);

  ASSERT_TRUE(vector.pop_back());
  ASSERT_EQ(construct_count, 1);
  ASSERT_EQ(destruct_count, 1);
  ASSERT_EQ(copy_count, 1);
}

TEST(TestFixedVector, HeapMovableType) {
  static size_t construct_count = 0;
  static size_t destruct_count = 0;
  static size_t move_count = 0;
  struct MovableType {
    MovableType() { construct_count++; }
    ~MovableType() { destruct_count++; }
    MovableType(const MovableType&) = delete;
    MovableType(MovableType&&) { move_count++; }
    MovableType& operator=(const MovableType&) = delete;
    MovableType& operator=(MovableType&&) = delete;
  };

  constexpr size_t kVectorSize = 1;
  FixedVector<MovableType> vector;

  ASSERT_TRUE(vector.reserve(kVectorSize));

  ASSERT_EQ(construct_count, 0);
  ASSERT_EQ(destruct_count, 0);
  ASSERT_EQ(move_count, 0);

  MovableType object;

  ASSERT_EQ(construct_count, 1);
  ASSERT_EQ(destruct_count, 0);
  ASSERT_EQ(move_count, 0);

  ASSERT_TRUE(vector.push_back(std::move(object)));
  ASSERT_EQ(construct_count, 1);
  ASSERT_EQ(destruct_count, 0);
  ASSERT_EQ(move_count, 1);

  ASSERT_TRUE(vector.pop_back());
  ASSERT_EQ(construct_count, 1);
  ASSERT_EQ(destruct_count, 1);
  ASSERT_EQ(move_count, 1);
}

TEST(TestFixedVector, HeapPointerType) {
  FixedVector<int*> vector;
  ASSERT_TRUE(vector.reserve(1));
  TestPointerType(vector);
}

TEST(TestFixedVector, HeapContiguousMemory) {
  FixedVector<uint8_t> vector;
  ASSERT_TRUE(vector.reserve(1024));
  TestContiguousMemory(vector);
}

TEST(TestFixedVector, HeapEmplace) {
  FixedVector<std::string> vector;
  ASSERT_TRUE(vector.reserve(10));
  TestEmplace(vector);
}

TEST(TestFixedVector, HeapErase) {
  FixedVector<std::string> vector;
  ASSERT_TRUE(vector.reserve(10));
  TestErase(vector);
}

TEST(TestFixedVector, HeapInsert) {
  FixedVector<std::string> vector;
  ASSERT_TRUE(vector.reserve(10));
  TestInsert(vector);
}

TEST(TestFixedVector, HeapResize) {
  struct DefaultType {
    DefaultType() = default;
    ~DefaultType() = default;
    DefaultType(const DefaultType&) = default;
    DefaultType(DefaultType&&) = default;
    DefaultType& operator=(const DefaultType&) = delete;
    DefaultType& operator=(DefaultType&&) = delete;
  };
  struct NonDefaultType {
    NonDefaultType(int) {}
    ~NonDefaultType() = default;
    NonDefaultType(const NonDefaultType&) = default;
    NonDefaultType(NonDefaultType&&) = default;
    NonDefaultType& operator=(const NonDefaultType&) = delete;
    NonDefaultType& operator=(NonDefaultType&&) = delete;
  };

  constexpr size_t kVectorSize = 1024;
  uint8_t data[kVectorSize];
  std::memset(data, 0xAA, kVectorSize);

  FixedVector<uint8_t> vector1;
  ASSERT_TRUE(vector1.reserve(kVectorSize));

  ASSERT_TRUE(vector1.resize(kVectorSize, 0xAA));

  ASSERT_EQ(vector1.size(), kVectorSize);
  ASSERT_EQ(std::memcmp(vector1.data(), data, kVectorSize), 0);

  ASSERT_TRUE(vector1.resize(kVectorSize / 2));

  ASSERT_EQ(vector1.size(), kVectorSize / 2);
  ASSERT_EQ(std::memcmp(vector1.data(), data, kVectorSize / 2), 0);

  FixedVector<DefaultType> vector2;
  ASSERT_TRUE(vector2.reserve(kVectorSize));

  ASSERT_EQ(vector2.size(), 0);
  ASSERT_TRUE(vector2.resize(kVectorSize));
  ASSERT_EQ(vector2.size(), kVectorSize);

  FixedVector<NonDefaultType> vector3;
  ASSERT_TRUE(vector3.reserve(kVectorSize));

  ASSERT_EQ(vector3.size(), 0);
  ASSERT_TRUE(vector3.resize(kVectorSize, NonDefaultType(0)));
  ASSERT_EQ(vector3.size(), kVectorSize);
}

TEST(TestFixedVector, HeapConst) {
  FixedVector<int> vector;
  ASSERT_TRUE(vector.reserve(1024));
  TestConst(vector);
}

TEST(TestFixedVector, HeapCopyFrom) {
  constexpr size_t kVectorSize = 1024;
  FixedVector<int> vector1;
  FixedVector<int> vector2;

  ASSERT_TRUE(vector1.reserve(kVectorSize));
  for (size_t i = 0; i < vector1.capacity(); i++) {
    ASSERT_TRUE(vector1.push_back(static_cast<int>(i)));
  }

  ASSERT_FALSE(vector2.copy_from(vector1));
  ASSERT_TRUE(vector2.reserve(kVectorSize));
  ASSERT_TRUE(vector2.copy_from(vector1));

  ASSERT_EQ(vector2.size(), kVectorSize);
  for (size_t i = 0; i < vector2.size(); i++) {
    auto element = vector1.at(i);
    ASSERT_TRUE(element);
    ASSERT_EQ(element.value(), static_cast<int>(i));
  }
}

TEST(TestFixedVector, HeapSubscriptOperator) {
  FixedVector<int> vector;
  ASSERT_TRUE(vector.reserve(16));
  TestSubscriptOperator(vector);
}

TEST(TestFixedVector, HeapIterator) {
  FixedVector<int> vector;
  ASSERT_TRUE(vector.reserve(1024));
  TestIterator(vector);
}

}  // namespace test
}  // namespace nvidia
