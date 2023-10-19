/*
Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gtest/gtest.h"

#include "common/unique_index_map.hpp"

#include <unordered_map>

namespace nvidia {
namespace gxf {

TEST(UniqueIndexMap, Construction) {
  class ExplicitInt {
   public:
    ExplicitInt() = delete;
    explicit ExplicitInt(int x) : x_(x) {}
   private:
    int x_;
  };

  // check initialized parameters are as expected
  UniqueIndexMap<int> bigMap;

  EXPECT_FALSE(bigMap.initialize(1ull << 33));
  EXPECT_TRUE(bigMap.initialize(1ull << 16));
  EXPECT_EQ(bigMap.capacity(), 1 << 16);
  EXPECT_EQ(bigMap.size(), 0);

  // make sure we track objects without default constructors
  UniqueIndexMap<ExplicitInt> noDefault;
  noDefault.initialize(16);
  EXPECT_EQ(noDefault.capacity(), 16);
  EXPECT_EQ(noDefault.size(), 0);
  EXPECT_TRUE(noDefault.insert(ExplicitInt(5)));
  EXPECT_EQ(noDefault.size(), 1);
}

TEST(UniqueIndexMap, Destruction) {
  static int constructCount = 0;
  static int destructCount = 0;
  struct DefaultOnly {
    DefaultOnly()  {
      ++constructCount;
    }

    ~DefaultOnly() {
      ++destructCount;
    }

    DefaultOnly(DefaultOnly&&) = delete;
    DefaultOnly(const DefaultOnly&) = delete;
    DefaultOnly& operator=(const DefaultOnly&) = delete;
    DefaultOnly& operator=(DefaultOnly&&) = delete;
  };

  {
    UniqueIndexMap<DefaultOnly> reg;
    EXPECT_TRUE(reg.initialize(10));
    EXPECT_EQ(constructCount, 0);
    EXPECT_EQ(destructCount, 0);

    for (int i = 0; i < 10; ++i) {
      EXPECT_TRUE(reg.emplace());
    }
    EXPECT_EQ(constructCount, 10);
    EXPECT_EQ(destructCount, 0);
  }

  EXPECT_EQ(destructCount, 10);
}

TEST(UniqueIndexMap, MoveContainerConstruction) {
  UniqueIndexMap<int> intMap;
  std::unordered_map<uint64_t, int> uids;

  EXPECT_TRUE(intMap.initialize(10));
  for (int i = 0; i < 10; ++i) {
    auto success = intMap.insert(i);
    EXPECT_TRUE(success);
    auto unique = uids.insert({success.value(), i});
    EXPECT_TRUE(unique.second);
  }

  EXPECT_EQ(intMap.size(), 10);
  EXPECT_EQ(intMap.capacity(), 10);

  UniqueIndexMap<int> intMap2(std::move(intMap));
  EXPECT_EQ(intMap.size(), 0);
  EXPECT_EQ(intMap.capacity(), 0);
  EXPECT_EQ(intMap2.size(), 10);
  EXPECT_EQ(intMap2.capacity(), 10);

  for (auto& kv : uids) {
    auto obj = intMap2.try_get(kv.first);
    EXPECT_TRUE(obj);
    EXPECT_EQ(*obj.value(), kv.second);
  }
}

TEST(UniqueIndexMap, MoveContainerAssignment) {
  UniqueIndexMap<int> intMap;
  std::unordered_map<uint64_t, int> uids;

  EXPECT_TRUE(intMap.initialize(10));
  for (int i = 0; i < 10; ++i) {
    auto success = intMap.insert(i);
    EXPECT_TRUE(success);
    auto unique = uids.insert({success.value(), i});
    EXPECT_TRUE(unique.second);
  }

  EXPECT_EQ(intMap.size(), 10);
  EXPECT_EQ(intMap.capacity(), 10);

  UniqueIndexMap<int> intMap2;
  intMap2 = std::move(intMap);
  EXPECT_EQ(intMap.size(), 0);
  EXPECT_EQ(intMap.capacity(), 0);
  EXPECT_EQ(intMap2.size(), 10);
  EXPECT_EQ(intMap2.capacity(), 10);

  for (auto& kv : uids) {
    auto obj = intMap2.try_get(kv.first);
    EXPECT_TRUE(obj);
    EXPECT_EQ(*obj.value(), kv.second);
  }
}

TEST(UniqueIndexMap, MoveInsertDelete) {
  static int constructCount = 0;
  static int destructCount = 0;
  struct MoveOnly {
    MoveOnly()  {
      ++constructCount;
    }

    MoveOnly(MoveOnly&&) {
      ++constructCount;
    }

    ~MoveOnly() {
      ++destructCount;
    }

    MoveOnly(const MoveOnly&) = delete;
    MoveOnly& operator=(const MoveOnly&) = delete;
    MoveOnly& operator=(MoveOnly&&) = delete;
  };

  EXPECT_EQ(constructCount, 0);
  UniqueIndexMap<MoveOnly> reg;
  EXPECT_TRUE(reg.initialize(16));
  EXPECT_EQ(constructCount, 0);
  EXPECT_EQ(destructCount, 0);

  // create the moved object at function scope to make lifetimes obvious
  MoveOnly counter{};
  EXPECT_EQ(constructCount, 1);
  EXPECT_EQ(destructCount, 0);

  auto insert_success = reg.insert(std::move(counter));
  EXPECT_TRUE(insert_success);
  EXPECT_EQ(constructCount, 2);
  EXPECT_EQ(destructCount, 0);

  EXPECT_TRUE(reg.erase(insert_success.value()));
  EXPECT_EQ(constructCount, 2);
  EXPECT_EQ(destructCount, 1);
}

TEST(UniqueIndexMap, CopyInsertDelete) {
  static int constructCount = 0;
  static int destructCount = 0;
  struct CopyOnly {
    CopyOnly()  {
      ++constructCount;
    }

    CopyOnly(const CopyOnly&) {
      ++constructCount;
    }

    ~CopyOnly() {
      ++destructCount;
    }

    CopyOnly(CopyOnly&&) = delete;
    CopyOnly& operator=(const CopyOnly&) = delete;
    CopyOnly& operator=(CopyOnly&&) = delete;
  };

  UniqueIndexMap<CopyOnly> reg;
  EXPECT_TRUE(reg.initialize(16));
  EXPECT_EQ(constructCount, 0);
  EXPECT_EQ(destructCount, 0);

  // create the copied object at function scope to make lifetimes obvious
  CopyOnly counter{};
  EXPECT_EQ(constructCount, 1);
  EXPECT_EQ(destructCount, 0);

  auto insert_success = reg.insert(counter);
  EXPECT_TRUE(insert_success);
  EXPECT_EQ(constructCount, 2);
  EXPECT_EQ(destructCount, 0);

  EXPECT_TRUE(reg.erase(insert_success.value()));
  EXPECT_EQ(constructCount, 2);
  EXPECT_EQ(destructCount, 1);
}

TEST(UniqueIndexMap, PointerStorage) {
  UniqueIndexMap<int*> ptrMap;
  EXPECT_TRUE(ptrMap.initialize(1));

  int a = 5;
  auto success = ptrMap.insert(&a);
  EXPECT_TRUE(success);

  auto obj = ptrMap.try_get(success.value());
  EXPECT_TRUE(obj);
  EXPECT_EQ(**obj.value(), 5);

  ++a;
  EXPECT_EQ(**obj.value(), 6);
}

TEST(UniqueIndexMap, Retrieve) {
  UniqueIndexMap<int> intMap;
  EXPECT_TRUE(intMap.initialize(16));
  std::unordered_map<uint64_t, int> uids;

  //  insert 10 elements into the map
  for (int i = 0; i < 10; ++i) {
    auto success = intMap.insert(int(i));
    EXPECT_TRUE(success);
    auto unique = uids.insert({success.value(), i});
    EXPECT_TRUE(unique.second);
  }
  EXPECT_EQ(intMap.size(), 10);
  EXPECT_EQ(uids.size(), 10);

  // make sure we get the same values back that we stored
  for (auto& kv : uids) {
    auto obj = intMap.try_get(kv.first);
    EXPECT_TRUE(obj);
    EXPECT_EQ(*obj.value(), kv.second);
  }
}

TEST(UniqueIndexMap, ConstRetrieve) {
  auto constGet = [](const UniqueIndexMap<int>& m, uint64_t id) -> const int*{
    auto result = m.try_get(id);
    return result ? result.value() : nullptr;
  };

  UniqueIndexMap<int> intMap;
  EXPECT_TRUE(intMap.initialize(1));

  auto success = intMap.insert(45);
  EXPECT_TRUE(success);

  auto constPtr = constGet(intMap, success.value());
  ASSERT_NE(constPtr, nullptr);
  EXPECT_EQ(*constPtr, 45);
}

TEST(UniqueIndexMap, InsertDelete) {
  UniqueIndexMap<int> intMap;
  std::unordered_map<uint64_t, int> uids;
  EXPECT_TRUE(intMap.initialize(10));

  // we can insert 10 elements without any problems
  for (int i = 0; i < 10; ++i) {
    auto success = intMap.insert(int(i));
    EXPECT_TRUE(success);
    auto unique = uids.insert({success.value(), i});
    EXPECT_TRUE(unique.second);
  }
  EXPECT_EQ(intMap.size(), 10);
  EXPECT_EQ(uids.size(), 10);

  // we are out of capacity, so insert should fail
  EXPECT_FALSE(intMap.insert(0));
  EXPECT_EQ(intMap.size(), 10);

  // remove all the previous ids
  for (auto& kv : uids) {
    EXPECT_TRUE(intMap.erase(kv.first)) << kv.first;
  }
  EXPECT_EQ(intMap.size(), 0);

  // double deleting all uids should fail
  for (auto& kv : uids) {
    EXPECT_FALSE(intMap.erase(kv.first));
  }
  EXPECT_EQ(intMap.size(), 0);

  // inserting 10 new ints should succeed, generating new uids
  for (int i = 0; i < 10; ++i) {
    auto success = intMap.insert(int(i));
    EXPECT_TRUE(success);
    auto unique = uids.insert({success.value(), i});
    EXPECT_TRUE(unique.second);
  }
  EXPECT_EQ(intMap.size(), 10);
  EXPECT_EQ(uids.size(), 20);
}

// We want to make sure that reusing the same internal memory address results in the correct return
// value with const and non-const members
TEST(UniqueIndexMap, AddressLaundering) {
  struct X {
    const int a;
    int b;
  };

  UniqueIndexMap<X> xMap;
  EXPECT_TRUE(xMap.initialize(1));

  auto success = xMap.emplace(3,4);
  EXPECT_TRUE(success);

  auto obj = xMap.try_get(success.value());
  EXPECT_TRUE(obj);
  EXPECT_EQ(obj.value()->a, 3);
  EXPECT_EQ(obj.value()->b, 4);

  EXPECT_TRUE(xMap.erase(success.value()));
  auto success2 = xMap.emplace(5,6);
  EXPECT_TRUE(success2);

  auto obj2 = xMap.try_get(success2.value());
  EXPECT_TRUE(obj2);
  EXPECT_EQ(obj.value()->a, 5);
  EXPECT_EQ(obj.value()->b, 6);
}

TEST(UniqueIndexMap, Find) {
  std::unordered_map<char, uint64_t> uids;
  UniqueIndexMap<char> map;
  ASSERT_TRUE(map.initialize(26));

  for (size_t i = 0; i < map.capacity(); i++) {
    const char c = 'a' + i;
    auto uid = map.insert(c);
    ASSERT_TRUE(uid);
    ASSERT_TRUE(map.find(c).assign_to(uids[c]));
    ASSERT_EQ(uid.value(), uids[c]);
  }

  ASSERT_TRUE(map.erase(uids['a']));
  EXPECT_FALSE(map.find('a'));
  EXPECT_TRUE(map.find('f'));
  EXPECT_TRUE(map.find('q'));
  EXPECT_TRUE(map.find('o'));
  EXPECT_TRUE(map.find('z'));

  ASSERT_TRUE(map.erase(uids['f']));
  EXPECT_FALSE(map.find('a'));
  EXPECT_FALSE(map.find('f'));
  EXPECT_TRUE(map.find('q'));
  EXPECT_TRUE(map.find('o'));
  EXPECT_TRUE(map.find('z'));

  ASSERT_TRUE(map.erase(uids['q']));
  EXPECT_FALSE(map.find('a'));
  EXPECT_FALSE(map.find('f'));
  EXPECT_FALSE(map.find('q'));
  EXPECT_TRUE(map.find('o'));
  EXPECT_TRUE(map.find('z'));

  ASSERT_TRUE(map.erase(uids['o']));
  EXPECT_FALSE(map.find('a'));
  EXPECT_FALSE(map.find('f'));
  EXPECT_FALSE(map.find('q'));
  EXPECT_FALSE(map.find('o'));
  EXPECT_TRUE(map.find('z'));

  ASSERT_TRUE(map.erase(uids['z']));
  EXPECT_FALSE(map.find('a'));
  EXPECT_FALSE(map.find('f'));
  EXPECT_FALSE(map.find('q'));
  EXPECT_FALSE(map.find('o'));
  EXPECT_FALSE(map.find('z'));
}

}  // namespace gxf
}  // namespace nvidia
