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
#include "gxf/sample/ping_rx.hpp"
#include "gxf/sample/ping_tx.hpp"
#include "gxf/std/block_memory_pool.hpp"
#include "gxf/std/tensor_copier.hpp"
#include "gxf/std/unbounded_allocator.hpp"
#include "gxf/test/components/tensor_comparator.hpp"
#include "gxf/test/components/tensor_generator.hpp"
#include "gxf/test/extensions/test_helpers.hpp"

namespace nvidia {
namespace gxf {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

class TestFactorialApp : public Application {
 public:
  void compose() override {
    // create an entity to get factorial
    auto gen = makeEntity<test::ArbitraryPrecisionFactorial>("gen",
                  makeTerm<CountSchedulingTerm>("count", Arg("count", 1)),
                  Arg("digits", 40000),
                  Arg("pool", makeResource<UnboundedAllocator>("pool")),
                  Arg("factorial", 10000));

    // create an entity to receive tensor
    auto printer = makeEntity<test::PrintTensor>("TensorReceiver");

    // add data flow connection gen -> printer
    connect(gen, printer);

    // add a scheduler component and configure the clock
    auto scheduler = setScheduler<Greedy>(Arg("max_duration_ms", 1000000));
  }
};

TEST(TestApp, TestFactorialApp) {
  auto app = create_app<TestFactorialApp>();
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_TRUE(app->loadExtensionManifest(kGxeManifestFilename).has_value());
  app->compose();
  GXF_ASSERT_TRUE(app->run().has_value());
}

}  // namespace gxf
}  // namespace nvidia
