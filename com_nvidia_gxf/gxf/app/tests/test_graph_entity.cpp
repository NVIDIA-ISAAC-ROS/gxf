/*
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <chrono>
#include <climits>
#include <cstring>
#include <iostream>
#include <thread>

#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/app/arg.hpp"
#include "gxf/app/graph_entity.hpp"
#include "gxf/core/gxf.h"
#include "gxf/sample/multi_ping_rx.hpp"
#include "gxf/sample/ping_rx.hpp"
#include "gxf/sample/ping_tx.hpp"
#include "gxf/std/clock.hpp"
#include "gxf/std/double_buffer_receiver.hpp"
#include "gxf/std/double_buffer_transmitter.hpp"
#include "gxf/std/scheduling_terms.hpp"

namespace {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

}  // namespace

namespace nvidia {
namespace gxf {

class GxfGraphEntityCreate : public ::testing::Test {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GxfSetSeverity(context, GXF_SEVERITY_VERBOSE);
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  }

  void TearDown() { GXF_ASSERT_SUCCESS(GxfContextDestroy(context)); }

 protected:
  gxf_context_t context = kNullContext;
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  const GxfEntityCreateInfo entity_create_info = {0};
};

template <typename T>
class GxfGraphEntityCreateT : public ::testing::TestWithParam<T> {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GxfSetSeverity(context, GXF_SEVERITY_VERBOSE);
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  }

  void TearDown() { GXF_ASSERT_SUCCESS(GxfContextDestroy(context)); }

 protected:
  gxf_context_t context = kNullContext;
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  const GxfEntityCreateInfo entity_create_info = {0};
};

TEST_F(GxfGraphEntityCreate, WrongName) {
  auto entity = GraphEntity();
  entity.setup(context, "basic");

  auto codelet = entity.addCodelet<PingTx>("PingTx");
  ASSERT_FALSE(codelet.is_null());

  auto wrong_tx = entity.addTransmitter<DoubleBufferTransmitter>("wrong_name");
  ASSERT_TRUE(wrong_tx.is_null());
}

TEST_F(GxfGraphEntityCreate, WrongType) {
  auto entity = GraphEntity();
  entity.setup(context, "basic");

  auto codelet = entity.addCodelet<PingTx>("PingTx");
  ASSERT_FALSE(codelet.is_null());

  auto wrong_rx = entity.addReceiver<DoubleBufferReceiver>("wrong_type");
  ASSERT_TRUE(wrong_rx.is_null());
}

class TestOmitTermVariant : public GxfGraphEntityCreateT<bool> {};

INSTANTIATE_TEST_CASE_P(Tensor, TestOmitTermVariant, ::testing::Values(false, true));

TEST_P(TestOmitTermVariant, AddTransmitter) {
  bool omit_term = GetParam();  // whether the default scheduling term will be omitted

  auto entity = GraphEntity();
  entity.setup(context, "basic");

  auto codelet = entity.addCodelet<PingTx>("PingTx");
  ASSERT_FALSE(codelet.is_null());

  auto tx = entity.addTransmitter<DoubleBufferTransmitter>(
      "signal", omit_term, Arg("policy", static_cast<uint64_t>(0)),
      Arg("capacity", static_cast<uint64_t>(5)));
  ASSERT_FALSE(tx.is_null());
  ASSERT_EQ(tx->getParameter<uint64_t>("capacity").value(), 5);
  ASSERT_EQ(tx->getParameter<uint64_t>("policy").value(), 0);
  if (!omit_term) {
    entity.configTransmitter("signal", 1, 2, 2);
    ASSERT_EQ(tx->getParameter<uint64_t>("capacity").value(), 1);
    ASSERT_EQ(tx->getParameter<uint64_t>("policy").value(), 2);
  }

  auto verify = entity.getTransmitter("signal");
  ASSERT_TRUE(verify);
  ASSERT_EQ(tx.cid(), verify.cid());

  auto verify2 = entity.get<DoubleBufferTransmitter>("signal");
  ASSERT_TRUE(verify2);
  ASSERT_EQ(tx.cid(), verify2.cid());

  auto verify3 = entity.get("nvidia::gxf::DoubleBufferTransmitter", "signal");
  ASSERT_TRUE(verify3);
  ASSERT_EQ(tx.cid(), verify3.cid());

  auto maybe_component = entity.try_get<nvidia::gxf::DoubleBufferTransmitter>("signal");
  ASSERT_TRUE(maybe_component);
  ASSERT_EQ(tx.cid(), maybe_component.value().cid());

  auto maybe_component2 = entity.try_get("nvidia::gxf::DoubleBufferTransmitter", "signal");
  ASSERT_TRUE(maybe_component2);
  ASSERT_EQ(tx.cid(), maybe_component2.value().cid());

  // check for non-existent component
  auto maybe_component3 = entity.try_get("nvidia::gxf::DoubleBufferReceiver");
  ASSERT_FALSE(maybe_component3);

  auto maybe_component4 = entity.try_get<nvidia::gxf::DoubleBufferReceiver>();
  ASSERT_FALSE(maybe_component4);

  auto result = codelet->getParameter<Handle<Transmitter>>("signal");
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value().cid(), tx.cid());

  auto term = entity.get<DownstreamReceptiveSchedulingTerm>("signal");
  if (omit_term) {
    ASSERT_TRUE(term.is_null());
  } else {
    ASSERT_FALSE(term.is_null());
    ASSERT_EQ(term->getParameter<uint64_t>("min_size").value(), 2);
  }

  auto periodic = entity.addSchedulingTerm<PeriodicSchedulingTerm>(
      "Periodic", Arg("recess_period", std::string("10ms")));
  ASSERT_FALSE(periodic.is_null());
  ASSERT_EQ(periodic->getParameter<std::string>("recess_period").value(), std::string("10ms"));

  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
}

TEST_P(TestOmitTermVariant, AddInvalidTerm) {
  auto entity = GraphEntity();
  entity.setup(context, "basic");

  auto invalidTerm = entity.addSchedulingTerm("Invalid", "Invalid", {});
  ASSERT_TRUE(invalidTerm.is_null());
}

TEST_P(TestOmitTermVariant, AddTransmitterViaTypeName) {
  bool omit_term = GetParam();  // whether the default scheduling term will be omitted

  auto entity = GraphEntity();
  entity.setup(context, "basic");

  auto codelet = entity.addCodelet("nvidia::gxf::PingTx", "PingTx");
  ASSERT_FALSE(codelet.is_null());

  std::vector<Arg> tx_args;
  tx_args.emplace_back(Arg("policy", static_cast<uint64_t>(0)));
  tx_args.emplace_back(Arg("capacity", static_cast<uint64_t>(5)));

  auto tx =
      entity.addTransmitter("nvidia::gxf::DoubleBufferTransmitter", "signal", tx_args, omit_term);
  ASSERT_FALSE(tx.is_null());
  ASSERT_EQ(tx->getParameter<uint64_t>("policy").value(), 0);
  ASSERT_EQ(tx->getParameter<uint64_t>("capacity").value(), 5);
  if (!omit_term) {
    entity.configTransmitter("signal", 1, 2, 2);
    ASSERT_EQ(tx->getParameter<uint64_t>("capacity").value(), 1);
    ASSERT_EQ(tx->getParameter<uint64_t>("policy").value(), 2);
  }

  auto verify = entity.getTransmitter("signal");
  ASSERT_TRUE(verify);
  ASSERT_EQ(tx.cid(), verify.cid());

  auto result = codelet->getParameter<Handle<Transmitter>>("signal");
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value().cid(), tx.cid());

  auto term = entity.get<DownstreamReceptiveSchedulingTerm>("signal");
  if (omit_term) {
    ASSERT_TRUE(term.is_null());
  } else {
    ASSERT_FALSE(term.is_null());
    ASSERT_EQ(term->getParameter<uint64_t>("min_size").value(), 2);
  }

  // add scheduling term via vector<Arg>
  std::vector<Arg> p_term_args;
  p_term_args.emplace_back(Arg("recess_period", std::string("10ms")));
  auto periodic =
      entity.addSchedulingTerm("nvidia::gxf::PeriodicSchedulingTerm", "Periodic", p_term_args);
  ASSERT_FALSE(periodic.is_null());
  ASSERT_EQ(periodic->getParameter<std::string>("recess_period").value(), std::string("10ms"));

  // add scheduling term with argument set via setParameter on Handle
  auto count = entity.addSchedulingTerm("nvidia::gxf::CountSchedulingTerm", "Count");
  ASSERT_FALSE(count.is_null());
  ASSERT_TRUE(count->setParameter<int64_t>("count", 10));

  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
}

TEST_P(TestOmitTermVariant, AddReceiver) {
  bool omit_term = GetParam();  // whether the default scheduling term will be omitted

  auto entity = GraphEntity();
  entity.setup(context, "basic");

  auto codelet = entity.addCodelet<PingRx>("PingRx");
  ASSERT_FALSE(codelet.is_null());

  auto rx = entity.addReceiver<DoubleBufferReceiver>("signal", omit_term,
                                                     Arg("policy", static_cast<uint64_t>(0)),
                                                     Arg("capacity", static_cast<uint64_t>(5)));
  ASSERT_FALSE(rx.is_null());
  ASSERT_EQ(rx->getParameter<uint64_t>("capacity").value(), 5);
  ASSERT_EQ(rx->getParameter<uint64_t>("policy").value(), 0);
  if (!omit_term) {
    entity.configReceiver("signal", 1, 2, 2);
    ASSERT_EQ(rx->getParameter<uint64_t>("capacity").value(), 1);
    ASSERT_EQ(rx->getParameter<uint64_t>("policy").value(), 2);
  }

  auto verify = entity.getReceiver("signal");
  ASSERT_TRUE(verify);
  ASSERT_EQ(rx.cid(), verify.cid());

  auto result = codelet->getParameter<Handle<Receiver>>("signal");
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value().cid(), rx.cid());

  auto term = entity.get<MessageAvailableSchedulingTerm>("signal");
  if (omit_term) {
    ASSERT_TRUE(term.is_null());
  } else {
    ASSERT_FALSE(term.is_null());
    ASSERT_EQ(term->getParameter<uint64_t>("min_size").value(), 2);
  }

  auto periodic = entity.addSchedulingTerm<PeriodicSchedulingTerm>("Periodic");
  ASSERT_FALSE(periodic.is_null());
  ASSERT_TRUE(periodic->setParameter<std::string>("recess_period", "10ms"));

  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
}

TEST_P(TestOmitTermVariant, AddReceiverViaTypeName) {
  bool omit_term = GetParam();  // whether the default scheduling term will be omitted

  auto entity = GraphEntity();
  entity.setup(context, "basic");

  auto codelet = entity.addCodelet("nvidia::gxf::PingRx", "PingRx");
  ASSERT_FALSE(codelet.is_null());

  std::vector<Arg> rx_args;
  rx_args.emplace_back(Arg("policy", static_cast<uint64_t>(0)));
  rx_args.emplace_back(Arg("capacity", static_cast<uint64_t>(5)));

  auto rx = entity.addReceiver("nvidia::gxf::DoubleBufferReceiver", "signal", rx_args, omit_term);
  ASSERT_FALSE(rx.is_null());
  ASSERT_EQ(rx->getParameter<uint64_t>("policy").value(), 0);
  ASSERT_EQ(rx->getParameter<uint64_t>("capacity").value(), 5);
  if (!omit_term) {
    entity.configReceiver("signal", 1, 2, 2);
    ASSERT_EQ(rx->getParameter<uint64_t>("capacity").value(), 1);
    ASSERT_EQ(rx->getParameter<uint64_t>("policy").value(), 2);
  }

  auto verify = entity.getReceiver("signal");
  ASSERT_TRUE(verify);
  ASSERT_EQ(rx.cid(), verify.cid());

  auto result = codelet->getParameter<Handle<Receiver>>("signal");
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value().cid(), rx.cid());

  auto term = entity.get<MessageAvailableSchedulingTerm>("signal");
  if (omit_term) {
    ASSERT_TRUE(term.is_null());
  } else {
    ASSERT_FALSE(term.is_null());
    ASSERT_EQ(term->getParameter<uint64_t>("min_size").value(), 2);
  }

  auto periodic = entity.addSchedulingTerm("nvidia::gxf::PeriodicSchedulingTerm", "Periodic",
                                           {Arg("recess_period", std::string("10ms"))});
  ASSERT_FALSE(periodic.is_null());
  ASSERT_EQ(periodic->getParameter<std::string>("recess_period").value(), std::string("10ms"));

  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
}

TEST_F(GxfGraphEntityCreate, FindAll) {
  auto entity = GraphEntity();
  entity.setup(context, "basic");

  auto tx_codelet = entity.addCodelet<PingTx>("PingTx");
  ASSERT_FALSE(tx_codelet.is_null());

  const auto& codelets = entity.findAll<Codelet>();
  ASSERT_EQ(codelets.capacity(), kMaxComponents);
  ASSERT_EQ(codelets.size(), 1);

  const auto& ping_tx_codelets = entity.findAll<PingTx, 1>();
  ASSERT_EQ(ping_tx_codelets.capacity(), 1);
  ASSERT_EQ(ping_tx_codelets.size(), 1);
}

TEST_F(GxfGraphEntityCreate, AddWithArgs) {
  auto entity = GraphEntity();
  entity.setup(context, "basic");

  auto clock = entity.add<RealtimeClock>("clock", Arg("initial_time_offset", 100.0),
                                         Arg("initial_time_scale", 2.0));
  ASSERT_FALSE(clock.is_null());
  ASSERT_EQ(clock->getParameter<double>("initial_time_offset").value(), 100.0);
  ASSERT_EQ(clock->getParameter<double>("initial_time_scale").value(), 2.0);
}

TEST_F(GxfGraphEntityCreate, AddClock) {
  auto entity = GraphEntity();
  entity.setup(context, "basic");

  auto clock = entity.addClock<RealtimeClock>("clock");
  ASSERT_FALSE(clock.is_null());
  ASSERT_TRUE(clock->setParameter<double>("initial_time_offset", 100.0));
  ASSERT_TRUE(clock->setParameter<double>("initial_time_scale", 2.0));
}

TEST_F(GxfGraphEntityCreate, AddNegativeClock) {
  auto entity = GraphEntity();
  entity.setup(context, "basic");

  auto clock = entity.addClock<RealtimeClock>("clock");
  ASSERT_FALSE(clock.is_null());

  ASSERT_FALSE(clock->sleepUntil(100 * 1000 * -1));
}

TEST_F(GxfGraphEntityCreate, GetClock) {
  auto entity = GraphEntity();
  entity.setup(context, "basic");

  auto clock = entity.addClock<RealtimeClock>("clock");
  ASSERT_FALSE(clock.is_null());

  auto dummyClock = entity.getClock("Dummy");
  ASSERT_TRUE(dummyClock.is_null());
}

TEST_F(GxfGraphEntityCreate, AddClockViaTypeName) {
  auto entity = GraphEntity();
  entity.setup(context, "basic");

  auto clock = entity.addClock("nvidia::gxf::RealtimeClock", "clock",
                               {Arg("initial_time_offset", 100.0), Arg("initial_time_scale", 2.0)});
  ASSERT_FALSE(clock.is_null());
  ASSERT_EQ(clock->getParameter<double>("initial_time_offset").value(), 100.0);
  ASSERT_EQ(clock->getParameter<double>("initial_time_scale").value(), 2.0);
}

TEST_F(GxfGraphEntityCreate, SetTimeScale) {
  auto entity = GraphEntity();
  entity.setup(context, "basic");

  auto clock = entity.addClock<RealtimeClock>("clock");
  ASSERT_FALSE(clock.is_null());

  ASSERT_FALSE(clock->setTimeScale(-1.0));
  ASSERT_TRUE(clock->setTimeScale(5.0));
}

TEST_F(GxfGraphEntityCreate, AddWithArgList) {
  auto entity = GraphEntity();
  entity.setup(context, "basic");

  std::vector<Arg> arg_list{Arg("initial_time_offset", 100.0), Arg("initial_time_scale", 2.0)};
  auto clock = entity.add<RealtimeClock>("clock", arg_list);
  ASSERT_FALSE(clock.is_null());
  ASSERT_EQ(clock->getParameter<double>("initial_time_offset").value(), 100.0);
  ASSERT_EQ(clock->getParameter<double>("initial_time_scale").value(), 2.0);
}

TEST_F(GxfGraphEntityCreate, VectorReceiver) {
  auto entity = GraphEntity();
  entity.setup(context, "basic");

  auto codelet = entity.addCodelet<MultiPingRx>("PingRx");
  ASSERT_FALSE(codelet.is_null());

  auto rx = entity.addReceiver<DoubleBufferReceiver>("receivers");
  ASSERT_FALSE(rx.is_null());
  ASSERT_EQ(std::string(rx->name()), std::string("receivers_0"));

  auto verify = entity.getReceiver("receivers_0");
  ASSERT_TRUE(verify);
  ASSERT_EQ(rx.cid(), verify.cid());

  auto result = codelet->getParameter<std::vector<Handle<Receiver>>>("receivers");
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value()[0].cid(), rx.cid());

  auto term = entity.get<MessageAvailableSchedulingTerm>("receivers_0");
  ASSERT_FALSE(term.is_null());

  auto rx2 = entity.addReceiver<DoubleBufferReceiver>("receivers");
  ASSERT_FALSE(rx2.is_null());
  ASSERT_EQ(std::string(rx2->name()), std::string("receivers_1"));

  auto verify2 = entity.getReceiver("receivers_1");
  ASSERT_TRUE(verify2);
  ASSERT_EQ(rx2.cid(), verify2.cid());

  result = codelet->getParameter<std::vector<Handle<Receiver>>>("receivers");
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value()[1].cid(), rx2.cid());

  auto term2 = entity.get<MessageAvailableSchedulingTerm>("receivers_0");
  ASSERT_FALSE(term2.is_null());

  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
}

TEST_F(GxfGraphEntityCreate, VectorReceiverViaTypeName) {
  auto entity = GraphEntity();
  entity.setup(context, "basic");

  auto codelet = entity.addCodelet("nvidia::gxf::MultiPingRx", "PingRx");
  ASSERT_FALSE(codelet.is_null());

  auto rx = entity.addReceiver("nvidia::gxf::DoubleBufferReceiver", "receivers");
  ASSERT_FALSE(rx.is_null());
  ASSERT_EQ(std::string(rx->name()), std::string("receivers_0"));

  auto verify = entity.getReceiver("receivers_0");
  ASSERT_TRUE(verify);
  ASSERT_EQ(rx.cid(), verify.cid());

  auto result = codelet->getParameter<std::vector<Handle<Receiver>>>("receivers");
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value()[0].cid(), rx.cid());

  auto term = entity.get<MessageAvailableSchedulingTerm>("receivers_0");
  ASSERT_FALSE(term.is_null());

  auto rx2 = entity.addReceiver("nvidia::gxf::DoubleBufferReceiver", "receivers");
  ASSERT_FALSE(rx2.is_null());
  ASSERT_EQ(std::string(rx2->name()), std::string("receivers_1"));

  auto verify2 = entity.getReceiver("receivers_1");
  ASSERT_TRUE(verify2);
  ASSERT_EQ(rx2.cid(), verify2.cid());

  result = codelet->getParameter<std::vector<Handle<Receiver>>>("receivers");
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value()[1].cid(), rx2.cid());

  auto term2 = entity.get<MessageAvailableSchedulingTerm>("receivers_0");
  ASSERT_FALSE(term2.is_null());

  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
}

TEST_F(GxfGraphEntityCreate, AddComponent) {
  auto entity = GraphEntity();
  entity.setup(context, "basic");

  auto component = entity.addComponent("nvidia::gxf::DummyComponent", "Dummy Component", {});
  ASSERT_TRUE(component.is_null());

  auto component1 =
      entity.addComponent("nvidia::gxf::DoubleBufferTransmitter", "Dummy Trasmitter", {});
  ASSERT_FALSE(component1.is_null());
}

}  // namespace gxf
}  // namespace nvidia
