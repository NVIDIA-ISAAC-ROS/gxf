/*
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gtest/gtest.h"

#include "gxf/core/expected.hpp"
#include "gxf/core/expected_macro.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace gxf {
namespace test {

namespace {

constexpr const char* kExtensions[] = {
  "gxf/std/libgxf_std.so",
};

constexpr GxfLoadExtensionsInfo kExtensionInfo = { kExtensions, 1, nullptr, 0, nullptr };
constexpr GxfEntityCreateInfo kEntityInfo = {"entity", 0};

}

template <typename T>
gxf::Expected<gxf::Handle<T>> add(gxf_context_t context,
                                  gxf_uid_t eid, const char* name) {
  gxf_tid_t tid;
  const gxf_result_t result_1 = GxfComponentTypeId(context, TypenameAsString<T>(), &tid);
  if (result_1 != GXF_SUCCESS) {
    return {Unexpected{result_1}, "Error Message Description for add - TypeId"};
  }

  gxf_uid_t cid;
  const gxf_result_t result_2 = GxfComponentAdd(context, eid, tid, name, &cid);
  if (result_2 != GXF_SUCCESS) {
    return {Unexpected{result_2}, "Error Message Description for add - Add"};
  }

  return gxf::Handle<T>::Create(context, cid);
}

template <typename T>
Expected<Handle<T>> get(gxf_context_t context, gxf_uid_t eid, const char* name = nullptr) {
  gxf_tid_t tid;
  const auto result_1 = GxfComponentTypeId(context, TypenameAsString<T>(), &tid);
  if (result_1 != GXF_SUCCESS) {
    return {Unexpected{result_1}, "Error Message Description for get - TypeId"};
  }

  gxf_uid_t cid;
  const auto result_2 = GxfComponentFind(context, eid, tid, name, nullptr, &cid);
  if (result_2 != GXF_SUCCESS) {
    return {Unexpected{result_2}, "Error Message Description for get - Find"};
  }

  return Handle<T>::Create(context, cid);
}

class TestExpectedMessage : public ::testing::Test {
 protected:
  void SetUp() {
    ASSERT_EQ(GxfContextCreate(&context_), GXF_SUCCESS);
    ASSERT_EQ(GxfLoadExtensions(context_, &kExtensionInfo), GXF_SUCCESS);
    ASSERT_EQ(GxfCreateEntity(context_, &kEntityInfo, &eid_), GXF_SUCCESS);
  }

  void TearDown() {
    ASSERT_EQ(GxfEntityDestroy(context_, eid_), GXF_SUCCESS);
    ASSERT_EQ(GxfContextDestroy(context_), GXF_SUCCESS);
  }

  gxf_context_t context_;
  gxf_uid_t eid_;
};

TEST_F(TestExpectedMessage, AddSuccess) {
  auto result =  add<gxf::Tensor>(context_, eid_, "Tensor");
  ASSERT_TRUE(result);
  ASSERT_TRUE(result.has_value());
}

TEST_F(TestExpectedMessage, GetSuccess) {
  auto result = add<gxf::Tensor>(context_, eid_, "Tensor");
  ASSERT_TRUE(result);
  ASSERT_TRUE(result.has_value());
  result = get<gxf::Tensor>(context_, eid_, "Tensor");
  ASSERT_TRUE(result);
  ASSERT_TRUE(result.has_value());
}

TEST_F(TestExpectedMessage, AddFailure) {
  auto result1 = add<gxf::Tensor>(nullptr, eid_, "Tensor");
  ASSERT_FALSE(result1);
  EXPECT_EQ(result1.get_error_message(), "Error Message Description for add - TypeId");

  auto result2 = add<gxf::Tensor>(context_, kNullUid, "Tensor");
  ASSERT_FALSE(result2);
  EXPECT_EQ(result2.get_error_message(), "Error Message Description for add - Add");
}

TEST_F(TestExpectedMessage, GetFailure) {
  auto result1 = get<gxf::Timestamp>(nullptr, eid_, "Unknown");
  ASSERT_FALSE(result1);
  EXPECT_EQ(result1.get_error_message(), "Error Message Description for get - TypeId");

  auto result2 = get<gxf::Timestamp>(context_, kNullUid, "Unknown");
  ASSERT_FALSE(result2);
  EXPECT_EQ(result2.get_error_message(), "Error Message Description for get - Find");
}

}  // namespace test
}  // namespace gxf
}  // namespace nvidia
