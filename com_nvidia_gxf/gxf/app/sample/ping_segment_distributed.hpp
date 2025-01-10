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
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>

#include "gxf/app/application.hpp"
#include "gxf/app/segment.hpp"
#include "gxf/sample/ping_tx.hpp"
#include "gxf/sample/ping_rx.hpp"

namespace nvidia {
namespace gxf {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest_extended.yaml";

class PingTxSegment : public Segment {
 public:
  void compose() override {
    // create a codelet to generate 10 messages
    auto tx_entity = makeEntity<PingTx>("Tx", makeTerm<CountSchedulingTerm>("count", Arg("count", 10)));

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

    // add a scheduler component and configure the clock
    auto scheduler = setScheduler<Greedy>(Arg("max_duration_ms", 5000), Arg("stop_on_deadlock", false));
  }
};

class SegmentDistributed : public Application {
 public:

  void compose() override {
    // segments plan
    auto tx_segment = createSegment<PingTxSegment>("TxSegment");
    auto rx_segment = createSegment<PingRxSegment>("RxSegment");

    // segments connection plan
    connect(tx_segment, rx_segment, {SegmentPortPair("Tx.signal", "Rx.signal")});
  }

};

}  // namespace gxf
}  // namespace nvidia
