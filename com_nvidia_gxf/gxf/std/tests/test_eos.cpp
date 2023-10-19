/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/eos.hpp"

#include "gtest/gtest.h"

namespace nvidia {
namespace gxf {

TEST(EndOfStream, createEoSMessage) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  constexpr const char* kExtensions[] = {
      "gxf/std/libgxf_std.so",
  };
  const GxfLoadExtensionsInfo info{kExtensions, 1, nullptr, 0, nullptr};

  ASSERT_EQ(GxfLoadExtensions(context, &info), GXF_SUCCESS);

  {
    Expected<Entity> maybe_entity = EndOfStream::createEoSMessage(context);
    ASSERT_TRUE(maybe_entity);

    Entity entity = maybe_entity.value();
    ASSERT_EQ(entity.findAll()->size(), 1);
    ASSERT_EQ(entity.findAll<EndOfStream>()->size(), 1);

    ASSERT_EQ(entity.findAll<EndOfStream>()->at(0).value()->stream_id(), -1);

    maybe_entity = EndOfStream::createEoSMessage(context, 4);
    ASSERT_TRUE(maybe_entity);

    entity = maybe_entity.value();
    ASSERT_EQ(entity.findAll()->size(), 1);
    ASSERT_EQ(entity.findAll<EndOfStream>()->size(), 1);

    ASSERT_EQ(entity.findAll<EndOfStream>()->at(0).value()->stream_id(), 4);
  }

  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

}  // namespace gxf
}  // namespace nvidia
