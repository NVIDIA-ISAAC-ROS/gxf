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

class PingMultiGatherApp : public Application {
 public:
  void compose() override {
    // create a codelet to generate 10 messages
    auto tx_1 = makeEntity<PingTx>("Tx_1",
                    makeTerm<PeriodicSchedulingTerm>("periodic", Arg("recess_period", "50Hz")),
                    makeTerm<CountSchedulingTerm>("count", Arg("count", 100)));

    auto tx_2 = makeEntity<PingTx>("Tx_2",
                    makeTerm<PeriodicSchedulingTerm>("periodic", Arg("recess_period", "50Hz")),
                    makeTerm<CountSchedulingTerm>("count", Arg("count", 100)));

    auto tx_3 = makeEntity<PingTx>("Tx_3",
                    makeTerm<PeriodicSchedulingTerm>("periodic", Arg("recess_period", "50Hz")),
                    makeTerm<CountSchedulingTerm>("count", Arg("count", 100)));

    auto forward = makeEntity<Forward>("Forward");

    // create a codelet to receive the messages
    auto ping_rx = makeEntity<PingRx>("PingRx");

    // add data flow connection tx -> rx
    connect(tx_1, forward);
    connect(tx_2, forward);
    connect(tx_3, forward);
    connect(forward, ping_rx);

    // configure the scheduler component
    setScheduler(SchedulerType::kGreedy);
  }
};

TEST(TestApp, PingMultiGatherApp) {
  auto app = create_app<PingMultiGatherApp>();
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_TRUE(app->loadExtensionManifest(kGxeManifestFilename).has_value());
  app->compose();
  GXF_ASSERT_TRUE(app->run().has_value());
}

}  // namespace gxf
}  // namespace nvidia
