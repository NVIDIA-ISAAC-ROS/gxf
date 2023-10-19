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
#include <vector>

#include "common/endian.hpp"

namespace nvidia {
namespace test {

namespace {

template <typename T>
void TestLittleEndian(const T value, const std::vector<uint8_t> encoding) {
  const T encoded = EncodeLittleEndian(value);
  const T decoded = DecodeLittleEndian(encoded);
  ASSERT_EQ(sizeof(encoded), encoding.size());
  ASSERT_EQ(std::memcmp(&encoded, encoding.data(), encoding.size()), 0);
  ASSERT_EQ(value, decoded);
}

template <typename T>
void TestBigEndian(const T value, const std::vector<uint8_t> encoding) {
  const T encoded = EncodeBigEndian(value);
  const T decoded = DecodeBigEndian(encoded);
  ASSERT_EQ(sizeof(encoded), encoding.size());
  ASSERT_EQ(std::memcmp(&encoded, encoding.data(), encoding.size()), 0);
  ASSERT_EQ(value, decoded);
}

}  // namespace

TEST(TestEndian, ByteOrder) {
  EXPECT_TRUE(IsLittleEndian());
  EXPECT_FALSE(IsBigEndian());
}

TEST(TestEndian, UInt64) {
  const uint64_t value = 0x0123456789ABCDEF;
  TestLittleEndian(value, { 0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01 });
  TestBigEndian(value, { 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF });
}

TEST(TestEndian, UInt32) {
  const uint32_t value = 0x01234567;
  TestLittleEndian(value, { 0x67, 0x45, 0x23, 0x01 });
  TestBigEndian(value, { 0x01, 0x23, 0x45, 0x67 });
}

TEST(TestEndian, UInt16) {
  const uint16_t value = 0x0123;
  TestLittleEndian(value, { 0x23, 0x01 });
  TestBigEndian(value, { 0x01, 0x23 });
}

TEST(TestEndian, UInt8) {
  const uint8_t value = 0x01;
  TestLittleEndian(value, { 0x01 });
  TestBigEndian(value, { 0x01 });
}

TEST(TestEndian, Int64) {
  const int64_t value = 0xFEDCBA9876543210;
  TestLittleEndian(value, { 0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE });
  TestBigEndian(value, { 0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10 });
}

TEST(TestEndian, Int32) {
  const int32_t value = 0xFEDCBA98;
  TestLittleEndian(value, { 0x98, 0xBA, 0xDC, 0xFE });
  TestBigEndian(value, { 0xFE, 0xDC, 0xBA, 0x98 });
}

TEST(TestEndian, Int16) {
  const int16_t value = 0xFEDC;
  TestLittleEndian(value, { 0xDC, 0xFE });
  TestBigEndian(value, { 0xFE, 0xDC });
}

TEST(TestEndian, Int8) {
  const int8_t value = 0xFE;
  TestLittleEndian(value, { 0xFE });
  TestBigEndian(value, { 0xFE });
}

}  // namespace test
}  // namespace nvidia
