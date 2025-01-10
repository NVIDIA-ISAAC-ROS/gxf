/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <chrono>
#include <climits>
#include <cstring>
#include <iostream>
#include <thread>

#include "gtest/gtest.h"

#include "gxf/app/application.hpp"
#include "gxf/app/graph_entity.hpp"
#include "gxf/sample/hello_world.hpp"

namespace nvidia {
namespace gxf {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

class HelloWorldSeg : public Segment {
 public:
  void compose() override {
    // create an entity to count 10 hello world messages
    auto helloworldentity = makeEntity<HelloWorld>("Hello world",
                                makeTerm<CountSchedulingTerm>("count", Arg("count", 10)));

    // add a scheduler component and configure the clock
    auto scheduler = setScheduler<Greedy>(Arg("stop_on_deadlock", false),
                                          Arg("max_duration_ms", 5000));
  }
};

class TestHelloWorldApp : public Application {
 public:
  void compose() override {
    // create a codelet to generate 10 messages
    auto graph_entity = createSegment<HelloWorldSeg>("HelloWorldSeg");
  }
};

TEST(TestApp, TestHelloWorldApp) {
  auto app = create_app<TestHelloWorldApp>();
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_TRUE(app->loadExtensionManifest(kGxeManifestFilename).has_value());
  app->compose();
  GXF_ASSERT_TRUE(app->run().has_value());
}

}  // namespace gxf
}  // namespace nvidia
