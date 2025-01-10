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
#include "gxf/std/forward.hpp"

namespace nvidia {
namespace gxf {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest_extended.yaml";

class PingTxSegment : public Segment {
 public:
  void compose() override {
    // add a clock component
    auto clock = setClock<RealtimeClock>("clock");

    // create a codelet to generate 10 messages
    auto tx_entity = makeEntity<PingTx>("Tx", makeTerm<CountSchedulingTerm>("count", Arg("count", int64_t(10))), Arg("clock", clock));

    // add a scheduler component and configure the clock
    auto scheduler = setScheduler<Greedy>(Arg("clock", clock), Arg("stop_on_deadlock", false),
                                          Arg("max_duration_ms", int64_t(10000)));
  }
};

class ForwardSegment : public Segment {
 public:
  void compose() override {
    // create a codelet to receive the messages
    auto fwd_entity = makeEntity<Forward>("Fwd");

    // add a scheduler component and configure the clock
    auto scheduler = setScheduler<Greedy>(Arg("max_duration_ms", int64_t(10000)),
                                          Arg("stop_on_deadlock", false));
  }
};

class PingRxSegment : public Segment {
 public:
  void compose() override {
    // create a codelet to receive the messages
    auto rx_entity = makeEntity<PingRx>("Rx");

    // add a scheduler component and configure the clock
    auto scheduler = setScheduler<Greedy>(Arg("max_duration_ms", int64_t(10000)),
                                          Arg("stop_on_deadlock", false));
  }
};

class SegmentDistributedGraph : public Application {
 public:

  void compose() override {
    // segments plan
    auto tx_segment = createSegment<PingTxSegment>("TxSegment");
    auto fwd_segment = createSegment<ForwardSegment>("FwdSegment");
    auto rx_segment = createSegment<PingRxSegment>("RxSegment");

    // segments connection plan
    connect(tx_segment, fwd_segment, {SegmentPortPair("Tx.signal", "Fwd.in")});
    connect(fwd_segment, rx_segment, {SegmentPortPair("Fwd.out", "Rx.signal")});
  }

};

}  // namespace gxf
}  // namespace nvidia
