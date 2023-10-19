/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gtest/gtest.h"

#include <functional>
#include <string>
#include <utility>

#include "common/fixed_string.hpp"

namespace nvidia {

namespace {

constexpr size_t kStringSizeNone = 0;
constexpr size_t kStringSizeTiny = 4;
constexpr size_t kStringSizeSmall = 16;
constexpr size_t kStringSizeMedium = 64;
constexpr size_t kStringSizeLarge = 256;
constexpr size_t kStringSizeMassive = 1024;

constexpr char kStringEmpty[] = "";
constexpr char kStringHelloWorld[] = "Hello, World!";
constexpr char kStringNumbers[] = "0123456789";
constexpr char kStringLetters[] = "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz";
constexpr char kStringSpecials[] = "~!@#$^&?.,:;_+-*/%=(){}[]<>";

}  // namespace

TEST(TestFixedString, DefaultConstruct) {
  FixedString<kStringSizeSmall> string;
  ASSERT_EQ(string, kStringEmpty);
}

TEST(TestFixedString, Null) {
  FixedString<kStringSizeNone> string;

  ASSERT_TRUE(string.empty());
  ASSERT_TRUE(string.full());

  ASSERT_TRUE(string.append(kStringEmpty));
  ASSERT_FALSE(string.append(kStringHelloWorld));

  ASSERT_TRUE(string.empty());
  ASSERT_TRUE(string.full());
}

TEST(TestFixedString, NoNullTerminator) {
  constexpr char array[] = {'a', 'b', 'c', 'd'};
  FixedString<kStringSizeTiny> string(array);

  ASSERT_TRUE(string.full());
  ASSERT_EQ(string, "abcd");
}

TEST(TestFixedString, StringOps) {
  FixedString<kStringSizeSmall> string(kStringHelloWorld);
  ASSERT_EQ(string, kStringHelloWorld);
  ASSERT_FALSE(string.empty());
  ASSERT_EQ(string.size(), string.length());
  ASSERT_EQ(string.max_size(), kStringSizeSmall);

  FixedString<kStringSizeMedium> string2(kStringNumbers);
  string2.copy(string);
  ASSERT_EQ(string2.compare(string), 0);
}

TEST(TestFixedString, CopyConstructCharacterArray) {
  FixedString<kStringSizeSmall> string(kStringHelloWorld);
  ASSERT_EQ(string, kStringHelloWorld);
}

TEST(TestFixedString, CopyConstructFixedString) {
  FixedString<kStringSizeSmall> string1(kStringHelloWorld);

  FixedString<kStringSizeSmall> string2(string1);
  ASSERT_EQ(string2, kStringHelloWorld);

  FixedString<kStringSizeMedium> string3(string2);
  ASSERT_EQ(string3, kStringHelloWorld);
}

TEST(TestFixedString, MoveConstructFixedString) {
  FixedString<kStringSizeSmall> string1(kStringHelloWorld);

  FixedString<kStringSizeSmall> string2(std::move(string1));
  ASSERT_EQ(string2, kStringHelloWorld);

  FixedString<kStringSizeMedium> string3(std::move(string2));
  ASSERT_EQ(string3, kStringHelloWorld);
}

TEST(TestFixedString, CopyAssignmentCharacterArray) {
  FixedString<kStringSizeSmall> string;
  string = kStringHelloWorld;
  ASSERT_EQ(string, kStringHelloWorld);
}

TEST(TestFixedString, CopyAssignmentFixedString) {
  FixedString<kStringSizeSmall> string1(kStringHelloWorld);

  FixedString<kStringSizeSmall> string2;
  string2 = string1;
  ASSERT_EQ(string2, kStringHelloWorld);

  FixedString<kStringSizeMedium> string3;
  string3 = string2;
  ASSERT_EQ(string3, kStringHelloWorld);
}

TEST(TestFixedString, MoveAssignmentFixedString) {
  FixedString<kStringSizeSmall> string1(kStringHelloWorld);

  FixedString<kStringSizeSmall> string2;
  string2 = std::move(string1);
  ASSERT_EQ(string2, kStringHelloWorld);

  FixedString<kStringSizeMedium> string3;
  string3 = std::move(string2);
  ASSERT_EQ(string2, kStringHelloWorld);
}

TEST(TestFixedString, EqualityCharacterArray) {
  FixedString<kStringSizeMedium> string(kStringLetters);
  ASSERT_NE(string, kStringEmpty);
  ASSERT_EQ(string, kStringLetters);
  ASSERT_NE(string, kStringNumbers);
}

TEST(TestFixedString, EqualityFixedString) {
  FixedString<kStringSizeMedium> string1(kStringLetters);
  FixedString<kStringSizeMedium> string2(kStringLetters);
  FixedString<kStringSizeMedium> string3(kStringNumbers);
  FixedString<kStringSizeLarge> string4(kStringLetters);
  ASSERT_EQ(string2, string1);
  ASSERT_NE(string3, string1);
  ASSERT_EQ(string4, string1);
}

TEST(TestFixedString, ComparisonCharacterArray) {
  FixedString<kStringSizeMedium> string(kStringLetters);
  ASSERT_LE(string, kStringLetters);
  ASSERT_GE(string, kStringLetters);
  ASSERT_GT(string, kStringEmpty);
  ASSERT_LT(string, kStringHelloWorld);
  ASSERT_GT(string, kStringNumbers);
  ASSERT_LT(string, kStringSpecials);
}

TEST(TestFixedString, ComparisonFixedString) {
  FixedString<kStringSizeMedium> string1(kStringLetters);
  FixedString<kStringSizeMedium> string2(kStringLetters);
  FixedString<kStringSizeMedium> string3(kStringEmpty);
  FixedString<kStringSizeMedium> string4(kStringHelloWorld);
  FixedString<kStringSizeMedium> string5(kStringNumbers);
  FixedString<kStringSizeMedium> string6(kStringSpecials);
  FixedString<kStringSizeLarge> string7(kStringLetters);
  ASSERT_LE(string2, string1);
  ASSERT_GE(string2, string1);
  ASSERT_LT(string3, string1);
  ASSERT_GT(string4, string1);
  ASSERT_LT(string5, string1);
  ASSERT_GT(string6, string1);
  ASSERT_LE(string7, string1);
  ASSERT_GE(string7, string1);
}

TEST(TestFixedString, AppendCString) {
  FixedString<kStringSizeMedium> actual(kStringEmpty);
  std::string expected;

  ASSERT_EQ(actual.length(), expected.length());
  ASSERT_EQ(expected.compare(actual.c_str()), 0);

  ASSERT_TRUE(actual.append(kStringNumbers, sizeof(kStringNumbers)));
  expected.append(kStringNumbers);

  ASSERT_EQ(actual.length(), expected.length());
  ASSERT_EQ(expected.compare(actual.c_str()), 0);

  ASSERT_TRUE(actual.append(kStringLetters, sizeof(kStringLetters)));
  expected.append(kStringLetters);

  ASSERT_EQ(actual.length(), expected.length());
  ASSERT_EQ(expected.compare(actual.c_str()), 0);

  ASSERT_FALSE(actual.append(kStringSpecials, sizeof(kStringSpecials)));

  ASSERT_EQ(actual.length(), expected.length());
  ASSERT_EQ(expected.compare(actual.c_str()), 0);

  ASSERT_FALSE(actual.append(nullptr, 0));

  ASSERT_EQ(actual.length(), expected.length());
  ASSERT_EQ(expected.compare(actual.c_str()), 0);
}

TEST(TestFixedString, AppendCharacterArray) {
  FixedString<kStringSizeMedium> actual(kStringEmpty);
  std::string expected;

  ASSERT_EQ(actual.length(), expected.length());
  ASSERT_EQ(expected.compare(actual.c_str()), 0);

  ASSERT_TRUE(actual.append(kStringNumbers));
  expected.append(kStringNumbers);

  ASSERT_EQ(actual.length(), expected.length());
  ASSERT_EQ(expected.compare(actual.c_str()), 0);

  ASSERT_TRUE(actual.append(kStringLetters));
  expected.append(kStringLetters);

  ASSERT_EQ(actual.length(), expected.length());
  ASSERT_EQ(expected.compare(actual.c_str()), 0);

  ASSERT_FALSE(actual.append(kStringSpecials));

  ASSERT_EQ(actual.length(), expected.length());
  ASSERT_EQ(expected.compare(actual.c_str()), 0);
}

TEST(TestFixedString, AppendFixedString) {
  FixedString<kStringSizeMedium> actual(kStringEmpty);
  std::string expected;

  ASSERT_EQ(actual.length(), expected.length());
  ASSERT_EQ(expected.compare(actual.c_str()), 0);

  ASSERT_TRUE(actual.append(FixedString<kStringSizeSmall>(kStringNumbers)));
  expected.append(kStringNumbers);

  ASSERT_EQ(actual.length(), expected.length());
  ASSERT_EQ(expected.compare(actual.c_str()), 0);

  ASSERT_TRUE(actual.append(FixedString<kStringSizeMedium>(kStringLetters)));
  expected.append(kStringLetters);

  ASSERT_EQ(actual.length(), expected.length());
  ASSERT_EQ(expected.compare(actual.c_str()), 0);

  ASSERT_FALSE(actual.append(FixedString<kStringSizeMedium>(kStringSpecials)));

  ASSERT_EQ(actual.length(), expected.length());
  ASSERT_EQ(expected.compare(actual.c_str()), 0);
}

TEST(TestFixedString, AppendCharacter) {
  FixedString<kStringSizeTiny> actual;
  std::string expected;

  ASSERT_EQ(actual.length(), expected.length());
  ASSERT_EQ(expected.compare(actual.c_str()), 0);

  ASSERT_TRUE(actual.append('A'));
  expected.append("A");

  ASSERT_EQ(actual.length(), expected.length());
  ASSERT_EQ(expected.compare(actual.c_str()), 0);

  ASSERT_TRUE(actual.append('B'));
  expected.append("B");

  ASSERT_EQ(actual.length(), expected.length());
  ASSERT_EQ(expected.compare(actual.c_str()), 0);

  ASSERT_TRUE(actual.append('C'));
  expected.append("C");

  ASSERT_EQ(actual.length(), expected.length());
  ASSERT_EQ(expected.compare(actual.c_str()), 0);

  ASSERT_TRUE(actual.append('D'));
  expected.append("D");

  ASSERT_EQ(actual.length(), expected.length());
  ASSERT_EQ(expected.compare(actual.c_str()), 0);

  ASSERT_FALSE(actual.append('E'));

  ASSERT_EQ(actual.length(), expected.length());
  ASSERT_EQ(expected.compare(actual.c_str()), 0);
}

TEST(TestFixedString, Iterator) {
  FixedString<kStringSizeMedium> string(kStringLetters);

  size_t count = 0;
  for (auto element : string) {
    ASSERT_TRUE(element);
    ASSERT_EQ(element.value(), kStringLetters[count++]);
  }

  ASSERT_EQ(count, string.size());
  ASSERT_EQ(string, kStringLetters);

  for (auto riter = string.rbegin(); riter != string.rend(); riter++) {
    auto element = *riter;
    ASSERT_TRUE(element);
    ASSERT_EQ(element.value(), kStringLetters[--count]);
  }

  ASSERT_EQ(count, 0);
  ASSERT_EQ(string, kStringLetters);

  auto begin = string.begin();
  auto end = string.end();

  ASSERT_TRUE(*begin);
  ASSERT_FALSE(*end);

  string.clear();

  ASSERT_FALSE(*begin);
  ASSERT_FALSE(*end);
}

TEST(TestFixedString, Hash) {
  FixedString<kStringSizeMedium> empty(kStringEmpty);
  FixedString<kStringSizeMedium> hello_world(kStringHelloWorld);
  FixedString<kStringSizeMedium> letters(kStringLetters);
  FixedString<kStringSizeMedium> numbers(kStringNumbers);
  FixedString<kStringSizeMedium> specials(kStringSpecials);

  EXPECT_EQ(FixedString<kStringSizeNone>::Hash{}(empty),
            std::hash<std::string>{}(empty.c_str()));
  EXPECT_EQ(FixedString<kStringSizeNone>::Hash{}(hello_world),
            std::hash<std::string>{}(hello_world.c_str()));
  EXPECT_EQ(FixedString<kStringSizeNone>::Hash{}(letters),
            std::hash<std::string>{}(letters.c_str()));
  EXPECT_EQ(FixedString<kStringSizeNone>::Hash{}(numbers),
            std::hash<std::string>{}(numbers.c_str()));
  EXPECT_EQ(FixedString<kStringSizeNone>::Hash{}(specials),
            std::hash<std::string>{}(specials.c_str()));
}

}  // namespace nvidia
