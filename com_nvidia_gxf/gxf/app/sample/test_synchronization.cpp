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
#include "gxf/sample/ping_tx.hpp"
#include "gxf/sample/ping_rx.hpp"
#include "gxf/std/gather.hpp"
#include "gxf/std/synchronization.hpp"
#include "gxf/std/tensor_copier.hpp"
#include "gxf/std/unbounded_allocator.hpp"
#include "gxf/test/components/tensor_comparator.hpp"
#include "gxf/test/components/tensor_generator.hpp"
#include "gxf/test/extensions/test_helpers.hpp"

namespace nvidia {
namespace gxf {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

class SynchronizationApp : public Application {
 public:
  void compose() override {

    // add a clock component
    auto clock = setClock<ManualClock>("clock");

    // create an entity to generate 5 messages
    auto tx1 = makeEntity<test::ScheduledPingTx>("Tx1",
                  makeTerm<CountSchedulingTerm>("count", Arg("count", 5)),
                  Arg("scheduling_term") = makeTerm<TargetTimeSchedulingTerm>("targettime", Arg("clock", clock)),
                  Arg("execution_clock", clock),
                  Arg("delay", 20000));
    tx1->add<test::StepCount>("StepCount", Arg("expected_count", 5));

    // create an entity to generate 10 messages
    auto tx2 = makeEntity<test::ScheduledPingTx>("Tx2",
                  makeTerm<CountSchedulingTerm>("count", Arg("count", 10)),
                  Arg("scheduling_term") = makeTerm<TargetTimeSchedulingTerm>("targettime", Arg("clock", clock)),
                  Arg("execution_clock", clock),
                  Arg("delay", 10000));
    tx2->add<test::StepCount>("StepCount", Arg("expected_count", 10));

    // create an entity to generate 20 messages
    auto tx3 = makeEntity<test::ScheduledPingTx>("Tx3",
                  makeTerm<CountSchedulingTerm>("count", Arg("count", 20)),
                  Arg("scheduling_term") = makeTerm<TargetTimeSchedulingTerm>("targettime", Arg("clock", clock)),
                  Arg("execution_clock", clock),
                  Arg("delay", 5000));
    tx3->add<test::StepCount>("StepCount", Arg("expected_count", 20));

    // create a synchronization entity
    auto sync = makeEntity<Synchronization>("Sync");

    auto gather = makeEntity<Gather>("Gather");

    auto pingrx = makeEntity<PingRx>("Rx");

    connect(tx1, sync, PortPair{"signal", "inputs"});
    connect(tx2, sync, PortPair{"signal", "inputs"});
    connect(tx3, sync, PortPair{"signal", "inputs"});
    connect(sync, gather, PortPair{"outputs", "sources"});
    connect(sync, gather, PortPair{"outputs", "sources"});
    connect(sync, gather, PortPair{"outputs", "sources"});
    connect(gather, pingrx);

    sync->configReceiver("inputs_0", 3, 2, 3);
    sync->configReceiver("inputs_1", 6, 2, 3);
    sync->configReceiver("inputs_2", 12, 2, 3);
    gather->configTransmitter("sink", 3, 2, 3);

    // add a scheduler component and configure the clock
    auto scheduler = setScheduler<Greedy>(Arg("clock", clock), Arg("max_duration_ms", 2000),
                                          Arg("stop_on_deadlock", true), Arg("check_recession_period_ms", 0),
                                          Arg("stop_on_deadlock_timeout", 0));
  }
};

TEST(TestApp, SynchronizationApp) {
  auto app = create_app<SynchronizationApp>();
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_TRUE(app->loadExtensionManifest(kGxeManifestFilename).has_value());
  app->compose();
  GXF_ASSERT_TRUE(app->run().has_value());
}

}  // namespace gxf
}  // namespace nvidia
