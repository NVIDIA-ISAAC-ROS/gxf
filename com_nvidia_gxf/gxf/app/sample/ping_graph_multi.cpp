/*
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gtest/gtest.h"

#include "common/assert.hpp"
#include "gxf/app/application.hpp"
#include "gxf/app/graph_entity.hpp"
#include "gxf/sample/multi_ping_rx.hpp"
#include "gxf/sample/ping_rx.hpp"
#include "gxf/sample/ping_tx.hpp"
#include "gxf/test/extensions/test_helpers.hpp"

namespace nvidia {
namespace gxf {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

class PingMultiApp : public Application {
 public:
  void compose() override {
    // create a codelet to generate 10 messages
    auto left_tx = makeEntity<PingTx>(
        "Left Tx", makeTerm<PeriodicSchedulingTerm>("periodic", Arg("recess_period", "50Hz")),
        makeTerm<CountSchedulingTerm>("count", Arg("count", 100)));

    auto right_tx = makeEntity<PingTx>(
        "Right Tx", makeTerm<PeriodicSchedulingTerm>("periodic", Arg("recess_period", "50Hz")),
        makeTerm<CountSchedulingTerm>("count", Arg("count", 100)));

    // create a codelet to receive the messages
    auto multi_ping_rx = makeEntity<MultiPingRx>("Multi Rx");

    // add data flow connection tx -> rx
    connect(left_tx, multi_ping_rx, PortPair{"signal", "receivers"});
    connect(right_tx, multi_ping_rx, PortPair{"signal", "receivers"});

    // configure the scheduler component
    setScheduler(SchedulerType::kGreedy);
  }
};

TEST(TestApp, PingMultiApp) {
  auto app = create_app<PingMultiApp>();
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_TRUE(app->loadExtensionManifest(kGxeManifestFilename).has_value());
  app->compose();
  GXF_ASSERT_TRUE(app->run().has_value());
}

}  // namespace gxf
}  // namespace nvidia
