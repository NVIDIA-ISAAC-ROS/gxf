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
#include "gxf/std/forward.hpp"

namespace nvidia {
namespace gxf {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

class PingMultiBroadcastApp : public Application {
 public:
  void compose() override {
    // create a codelet to generate 10 messages
    auto tx = makeEntity<PingTx>("Tx_1", makeTerm<PeriodicSchedulingTerm>
                                         ("periodic", Arg("recess_period", "50Hz")),
                                         makeTerm<CountSchedulingTerm>("count", Arg("count", 100)));

    auto forward = makeEntity<Forward>("Forward");

    // create a codelet to receive the messages
    auto rx_1 = makeEntity<PingRx>("Rx_1");
    auto rx_2 = makeEntity<PingRx>("Rx_2");
    auto rx_3 = makeEntity<PingRx>("Rx_3");

    // add data flow connection tx -> rx
    connect(tx, forward);
    connect(forward, rx_1);
    connect(forward, rx_2);
    connect(forward, rx_3);

    // configure the scheduler component
    setScheduler(SchedulerType::kGreedy);
  }
};

TEST(TestApp, PingMultiBroadcastApp) {
  auto app = create_app<PingMultiBroadcastApp>();
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_TRUE(app->loadExtensionManifest(kGxeManifestFilename).has_value());
  app->compose();
  GXF_ASSERT_TRUE(app->run().has_value());
}

}  // namespace gxf
}  // namespace nvidia
