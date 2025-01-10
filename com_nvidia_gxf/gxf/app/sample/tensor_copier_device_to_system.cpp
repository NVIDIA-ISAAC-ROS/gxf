/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <vector>

#include "gtest/gtest.h"

#include "common/assert.hpp"
#include "gxf/app/application.hpp"
#include "gxf/app/graph_entity.hpp"
#include "gxf/sample/ping_rx.hpp"
#include "gxf/sample/ping_tx.hpp"
#include "gxf/std/tensor_copier.hpp"
#include "gxf/std/unbounded_allocator.hpp"
#include "gxf/test/components/tensor_comparator.hpp"
#include "gxf/test/components/tensor_generator.hpp"
#include "gxf/test/extensions/test_helpers.hpp"

namespace nvidia {
namespace gxf {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

class DeviceToSystemApp : public Application {
 public:
  void compose() override {
    // create an entity to generate 10 messages on device
    auto generator_device = makeEntity<test::TensorGenerator>("Generator_device",
                                makeTerm<CountSchedulingTerm>("count", Arg("count", 10)),
                                Arg("enable_timestamps", false), Arg("storage_type", 1),
                                Arg("shape", std::vector<int32_t>{4, 4, 4, 4}),
                                Arg("allocator") = makeResource<UnboundedAllocator>("allocator"));

    // create an entity to copy messages from device to host
    auto copier = makeEntity<TensorCopier>("Copier", Arg("allocator") =
                      makeResource<UnboundedAllocator>("allocator"), Arg("mode", 2));

    // create an entity to generate 10 messages on system
    auto generator_system = makeEntity<test::TensorGenerator>("Generator_host",
                                makeTerm<CountSchedulingTerm>("count", Arg("count", 10)),
                                Arg("enable_timestamps", false), Arg("storage_type", 2),
                                Arg("shape", std::vector<int32_t>{4, 4, 4, 4}),
                                Arg("allocator") = makeResource<UnboundedAllocator>("allocator"));

    auto comparator = makeEntity<test::TensorComparator>("Comparator");
    comparator->add<test::StepCount>("Count_device", Arg("expected_count", 10));

    // add data flow connection device -> system
    connect(generator_device, copier);
    connect(copier, comparator, PortPair{"transmitter", "actual"});
    connect(generator_system, comparator, PortPair{"output", "expected"});

    // add a scheduler component and auto configure the clock
    setScheduler<Greedy>(Arg("max_duration_ms", 1000));
  }
};

TEST(TestApp, DeviceToSystemApp) {
  auto app = create_app<DeviceToSystemApp>();
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_TRUE(app->loadExtensionManifest(kGxeManifestFilename).has_value());
  app->compose();
  GXF_ASSERT_TRUE(app->run().has_value());
}

}  // namespace gxf
}  // namespace nvidia
