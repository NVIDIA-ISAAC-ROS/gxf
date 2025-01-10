/*
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <chrono>
#include <cstring>
#include <thread>
#include <iostream>
#include <fstream>
#include <sstream>

#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"
#include "gxf/core/entity.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/std/default_extension.hpp"
#include "gxf/std/gems/utils/time.hpp"

#include "gxf/test/unit/test_load_extension.hpp"

namespace {

constexpr const char* kStdExtensionFilename = "gxf/std/libgxf_std.so";

}  // namespace

namespace nvidia {
namespace gxf {

TEST(Entity, MT) {
  gxf_context_t context;
  GxfContextCreate(&context);
  const GxfLoadExtensionsInfo info{&kStdExtensionFilename, 1, nullptr, 0, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));

  const int kNumThread = 1;
  const int kEntitiesPerThread = 100000;  // 100k per thread
  const int kCompPerEntity = 10;
  std::thread t[kNumThread];

  // Time to start creating 100k Entity objects per thread
  long timeBegin = getCurrentTimeUs();
  (void)timeBegin;  // avoid unused variable warning
  for (int j = 0; j < kNumThread; j++) {
    t[j] = std::thread([context, j] () mutable {
      for (int i = 0; i < kEntitiesPerThread; i++) {
        Entity ent;
        ent = std::move(Entity::New(context).value());
        // add 10 components of type int per Entity
        for (int j = 0; j < kCompPerEntity; j++) {
          ent.add<int>();
        }
      }
    });
  }
  for (int j = 0; j < kNumThread; j++) {
    t[j].join();
  }
  // Time to finish all threads
  long timeEnd = getCurrentTimeUs();
  (void)timeEnd;  // avoid unused variable warning
  GXF_LOG_INFO("**********************TIME TAKEN %ld ms, threadNum: %d\n",
    (timeEnd - timeBegin)/1000, kNumThread);

  GxfContextDestroy(context);
}
}
}
