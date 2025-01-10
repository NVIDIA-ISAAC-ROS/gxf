/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gtest/gtest.h"

#include "gxf/app/application.hpp"
#include "gxf/app/graph_entity.hpp"
#include "gxf/sample/hello_world.hpp"

namespace nvidia {
namespace gxf {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

class HelloWorldMultiSeg : public Segment {
 public:
  void compose() override {
    // create a codelet to count 10 hello world messages
    auto helloworldentity = makeEntity<HelloWorld>("Hello world",
                                makeTerm<CountSchedulingTerm>("count", Arg("count", 10)));

    // adds a scheduler component and configure the clock
    auto scheduler = setScheduler<MultiThread>(Arg("stop_on_deadlock", true),
                                               Arg("max_duration_ms", 1000));
  }
};

class TestHelloWorldMultiApp : public Application {
 public:
  void compose() override {
    // create a codelet to generate 10 messages
    auto graph_entity = createSegment<HelloWorldMultiSeg>("HelloWorldSegment");
  }
};

TEST(TestApp, TestHelloWorldMultiApp) {
  auto app = create_app<TestHelloWorldMultiApp>();
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_TRUE(app->loadExtensionManifest(kGxeManifestFilename).has_value());
  app->compose();
  GXF_ASSERT_TRUE(app->run().has_value());
}

}  // namespace gxf
}  // namespace nvidia
