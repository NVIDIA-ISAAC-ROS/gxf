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
#include "gxf/sample/ping_rx.hpp"
#include "gxf/sample/ping_tx.hpp"

namespace nvidia {
namespace gxf {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

class PingTxRuntime : public Codelet {
 public:
  virtual ~PingTxRuntime() = default;

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(signal_, "signal", "Signal",
              "Transmitter channel publishing messages to other graph entities");
    return ToResultCode(result);
  }

  gxf_result_t tick() override {
    auto message = Entity::New(context());
    if (!message) {
      GXF_LOG_ERROR("Failure creating message entity.");
      return message.error();
    }

    auto result = signal_->publish(message.value());
    GXF_LOG_INFO("Message Sent: %d", this->count);
    this->count = this->count + 1;
    return ToResultCode(message);
  }

 private:
  Parameter<Handle<Transmitter>> signal_;
  int count = 1;
};

class PingRuntimeApp : public Application {
 public:
  void compose() override {
    // create a codelet to generate 10 messages
    auto tx_entity = makeEntity<PingTxRuntime>("Tx",
                        makeTerm<CountSchedulingTerm>("count", Arg("count", 10)));

    // create a codelet to receive the messages
    auto rx_entity = makeEntity<PingRx>("Rx");

    // add data flow connection tx -> rx
    connect(tx_entity, rx_entity);

    // configure the scheduler component
    setScheduler(SchedulerType::kGreedy);
  }
};

TEST(TestApp, PingRuntimeApp) {
  auto app = create_app<PingRuntimeApp>();
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_TRUE(app->loadExtensionManifest(kGxeManifestFilename).has_value());
  app->compose();
  GXF_ASSERT_TRUE(app->run().has_value());
}

}  // namespace gxf
}  // namespace nvidia
