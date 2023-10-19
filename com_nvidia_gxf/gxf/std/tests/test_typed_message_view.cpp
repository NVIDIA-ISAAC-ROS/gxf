/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <random>

#include "gtest/gtest.h"
#include "gxf/core/gxf.h"
#include "common/assert.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/transmitter.hpp"
#include "gxf/core/component.hpp"

#include "gxf/std/typed_message_view.hpp"
#include <stdio.h>
#include <iostream>

#include "gxf/std/tests/test_typed_message_view_helper.hpp"

namespace nvidia {
namespace gxf {

TEST(TypedMessageView, addingTensorToMessage) {
  constexpr uint64_t kBlockSize = 1024;
  constexpr uint64_t kNumBlocks = 1;


  gxf_context_t current_context;
  ASSERT_EQ(GxfContextCreate(&current_context), GXF_SUCCESS);

  constexpr const char* kExtensions[] = {
      "gxf/std/libgxf_std.so",
  };

  const GxfLoadExtensionsInfo info{kExtensions, 1, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(current_context, &info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(current_context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(current_context, "nvidia::gxf::BlockMemoryPool", &tid), GXF_SUCCESS);

  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(current_context, eid, tid, "test", &cid), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(current_context, cid, tid, &pointer), GXF_SUCCESS);

  ASSERT_EQ(GxfParameterSetUInt64(current_context, cid, "block_size", kBlockSize), GXF_SUCCESS);
  ASSERT_EQ(GxfParameterSetUInt64(current_context, cid, "num_blocks", kNumBlocks), GXF_SUCCESS);

  auto maybe_allocator = Handle<Allocator>::Create(current_context, cid);
  ASSERT_TRUE(maybe_allocator.has_value());

  auto allocator = maybe_allocator.value();

  ASSERT_EQ(allocator->initialize(), GXF_SUCCESS);

  ASSERT_EQ(allocator->is_available_abi(kBlockSize), GXF_SUCCESS);

  Expected<Entity> my_msg = Entity::New(current_context);
  ASSERT_TRUE(my_msg.has_value());

  nvidia::gxf::TypedMessageView<Tensor> test_format_1("tensor1");

  test_format_1.add_to_entity(my_msg.value());

  auto frame = my_msg.value().get<Tensor>("tensor1");
  auto NOTE = my_msg.value().get<Tensor>("NOT_HERE");

  decltype(get_ele<0>(test_format_1.format_))::comp_type_ frame_tensor {};

  bool test = std::is_same<decltype(get_ele<0>(test_format_1.format_))::comp_type_, Tensor>::value;
  ASSERT_TRUE(test);

  test_format_1.check_entity(my_msg.value());

  nvidia::gxf::TypedMessageView<Tensor> audio("captions");

  frame_tensor.reshapeCustom(Shape({1, kBlockSize}), PrimitiveType::kUnsigned8, 1,
                        Unexpected{GXF_UNINITIALIZED_VALUE}, MemoryStorageType::kHost, allocator);

  ASSERT_EQ(allocator->is_available_abi(kBlockSize), GXF_FAILURE);

  auto result = audio.check_entity(my_msg.value());
  ASSERT_EQ(result, GXF_FAILURE); // this should fail because we have not done add_to_entity with audio

  // testing with format in test_message_format_helper.hpp

  nvidia::gxf::my_formats::test_format_2.add_to_entity(my_msg.value());

  my_msg.value().add<Tensor>("tensor33");
  auto check_res = my_msg.value().get<Tensor>("ten");
  ASSERT_TRUE(my_msg.has_value());

  // TypedMessageViews with reverse types are independent objects
  nvidia::gxf::TypedMessageView<Tensor, int> forward("T5", "my_int");
  nvidia::gxf::TypedMessageView<int, Tensor> reverse("my_other_int", "T6");

  forward.add_to_entity(my_msg.value());
  reverse.add_to_entity(my_msg.value());

  ASSERT_EQ(allocator->deinitialize(), GXF_SUCCESS);

}


}  // namespace gxf
}  // namespace nvidia
