/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "common/type_name.hpp"

#include "gtest/gtest.h"

struct Foo { };

TEST(TypenameAsString, Test1) {
  ASSERT_STREQ(nvidia::TypenameAsString<Foo>(), "Foo");
}

namespace hussa {

struct Foo { };

TEST(TypenameAsString, Test2a) {
  ASSERT_STREQ(nvidia::TypenameAsString<Foo>(), "hussa::Foo");
}

TEST(TypenameAsString, Test2b) {
  ASSERT_STREQ(nvidia::TypenameAsString<::Foo>(), "Foo");
}

}  // namespace hussa

namespace hussa {
namespace bravo {

struct Foo { };

TEST(TypenameAsString, Test3a) {
  ASSERT_STREQ(nvidia::TypenameAsString<Foo>(), "hussa::bravo::Foo");
}

TEST(TypenameAsString, Test3b) {
  ASSERT_STREQ(nvidia::TypenameAsString<::hussa::Foo>(), "hussa::Foo");
}

TEST(TypenameAsString, Test3c) {
  ASSERT_STREQ(nvidia::TypenameAsString<::Foo>(), "Foo");
}

}  // namespace bravo
}  // namespace hussa

namespace bravo {

struct Foo { };

TEST(TypenameAsString, Test4a) {
  ASSERT_STREQ(nvidia::TypenameAsString<Foo>(), "bravo::Foo");
}

TEST(TypenameAsString, Test4b) {
  ASSERT_STREQ(nvidia::TypenameAsString<::hussa::Foo>(), "hussa::Foo");
}

TEST(TypenameAsString, Test4c) {
  ASSERT_STREQ(nvidia::TypenameAsString<::Foo>(), "Foo");
}

}  // namespace bravo

namespace hussa {
namespace bravo {

TEST(TypenameAsString, Test5a) {
  ASSERT_STREQ(nvidia::TypenameAsString<Foo>(), "hussa::bravo::Foo");
}

TEST(TypenameAsString, Test5b) {
  ASSERT_STREQ(nvidia::TypenameAsString<::bravo::Foo>(), "bravo::Foo");
}

TEST(TypenameAsString, Test5c) {
  ASSERT_STREQ(nvidia::TypenameAsString<bravo::Foo>(), "hussa::bravo::Foo");
}

}  // namespace bravo
}  // namespace hussa

namespace VeryLongNamespaceVeryLongNamespaceVeryLongNamespaceVeryLongNamespaceVeryLongNamespace {
namespace VeryLongNamespaceVeryLongNamespaceVeryLongNamespaceVeryLongNamespaceVeryLongNamespace {
namespace VeryLongNamespaceVeryLongNamespaceVeryLongNamespaceVeryLongNamespaceVeryLongNamespace {

struct AbsurdlyLongTypenameAbsurdlyLongTypenameAbsurdlyLongTypename;

TEST(TypenameAsString, Test6) {
  ASSERT_STREQ(
      nvidia::TypenameAsString<AbsurdlyLongTypenameAbsurdlyLongTypenameAbsurdlyLongTypename>(),
      "VeryLongNamespaceVeryLongNamespaceVeryLongNamespaceVeryLongNamespaceVeryLongNamespace::"
      "VeryLongNamespaceVeryLongNamespaceVeryLongNamespaceVeryLongNamespaceVeryLongNamespace::"
      "VeryLongNamespaceVeryLongNamespaceVeryLongNamespaceVeryLongNamespaceVeryLongNamespace::"
      "AbsurdlyLongTypenameAbsurdlyLongTypenameAbsurdlyLongTypename");
}

}  // namespace
}  // namespace
}  // namespace
