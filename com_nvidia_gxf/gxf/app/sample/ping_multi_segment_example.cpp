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
#include "gxf/app/segment.hpp"
#include "gxf/sample/ping_rx.hpp"
#include "gxf/sample/ping_tx.hpp"

namespace nvidia {
namespace gxf {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

class PingTxSegment : public Segment {
 public:
  void compose() override {
    // create a codelet to generate 10 messages
    auto tx_entity = makeEntity<PingTx>("Tx",
                        makeTerm<CountSchedulingTerm>("count", Arg("count", 10)));

    // add a scheduler component and configure the clock
    auto scheduler = setScheduler<Greedy>(Arg("stop_on_deadlock", false),
                                          Arg("max_duration_ms", 5000));
  }
};

class PingRxSegment : public Segment {
 public:
  void compose() override {
    // create a codelet to receive the messages
    auto rx_entity = makeEntity<PingRx>("Rx");

    // add a clock component
    auto clock = setClock<RealtimeClock>("clock");

    // add a scheduler component and configure the clock
    auto scheduler = setScheduler<Greedy>(Arg("max_duration_ms", 5000),
                                          Arg("stop_on_deadlock", false));
  }
};


class PingMultiSegmentApp : public Application {
 public:
  void compose() override {
    auto tx1_segment = createSegment<PingTxSegment>("TxSegment1");
    auto rx1_segment = createSegment<PingRxSegment>("RxSegment1");

    auto tx2_segment = createSegment<PingTxSegment>("TxSegment2");
    auto rx2_segment = createSegment<PingRxSegment>("RxSegment2");


    // add data flow connection tx -> rx
    connect(tx1_segment, rx1_segment, {SegmentPortPair{"Tx.signal", "Rx.signal"}});
    connect(tx2_segment, rx2_segment, {SegmentPortPair("Tx.signal", "Rx.signal")});
  }
};

TEST(TestApp, PingMultiSegmentApp) {
  auto app = create_app<PingMultiSegmentApp>();
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_TRUE(app->loadExtensionManifest(kGxeManifestFilename).has_value());
  app->compose();
  GXF_ASSERT_TRUE(app->run().has_value());
}

}  // namespace gxf
}  // namespace nvidia
