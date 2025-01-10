/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <string>
#include "gtest/gtest.h"

#include "gxf/app/sample/ping_segment_distributed_graph.hpp"
#include "common/assert.hpp"

namespace {
std::string getExecutablePath() {
#if defined(__linux__)
  return std::filesystem::canonical("/proc/self/exe").parent_path().string();
#else
  return std::filesystem::canonical(std::filesystem::path(__FILE__).parent_path()).string();
#endif
}
}  // namespace
namespace nvidia {
namespace gxf {

constexpr const char* kGxeManifestExtFilename = "gxf/gxe/manifest_extended.yaml";
constexpr const char* kInvalidGxFManifestFilename = "gxf/gxe/tests/manifest_invalid.yaml";
constexpr const char* kEmptyGxFManifestFilename = "gxf/gxe/tests/manifest_empty.yaml";
constexpr const char* kGxeConfigFilename = "gxf/test/apps/ping_segment_distributed_graph_w3.yaml";

class SegmentDistributedGraphValidTxInvalidRx : public Application {
 public:
  void compose() override {
    // segments plan
    auto tx_segment = createSegment<PingTxSegment>("TxSegment");
    auto fwd_segment = createSegment<ForwardSegment>("FwdSegment");
    auto rx_segment = createSegment<PingRxSegment>("RxSegment");

    // segments connection plan
    connect(tx_segment, nullptr, {SegmentPortPair("Tx.signal", "Fwd.in")});
    connect(nullptr, rx_segment, {SegmentPortPair("Fwd.out", "Rx.signal")});
  }
};

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

TEST(TestApp, InvalidExtensionManifestFile) {
  auto app = create_app<PingMultiBroadcastApp>();
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_FALSE(app->loadExtensionManifest("").has_value());
}

TEST(TestApp, InvalidConnection) {
  auto app = create_app<SegmentDistributedGraphValidTxInvalidRx>();
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_FALSE(app->loadExtensionManifest("").has_value());
}

TEST(TestApp, InvalidExtensionInfo) {
  auto app = create_app<PingMultiBroadcastApp>();
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_FALSE(app->loadExtensionManifest("").has_value());
}

TEST(TestApp, InvalidConfig) {
  auto app = create_app<PingMultiBroadcastApp>();
  app->setConfig("");
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_FALSE(app->loadExtensionManifest("").has_value());
}

TEST(TestApp, ValidConfigFilePath) {
  auto app = create_app<SegmentDistributedGraph>();
  GXF_ASSERT_TRUE(app->setConfig(kGxeConfigFilename).has_value());
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_TRUE(app->loadExtensionManifest(kGxeManifestExtFilename).has_value());
  app->compose();
  app->runAsync();
  app->interrupt();
}

TEST(TestApp, ValidConfigWithArguments) {
  auto app = create_app<SegmentDistributedGraph>();
  int argc = 2;
  char* argv[] = {const_cast<char*>("TestApp"), const_cast<char*>(kGxeConfigFilename)};

  GXF_ASSERT_TRUE(app->setConfig(argc, argv).has_value());
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_TRUE(app->loadExtensionManifest(kGxeManifestExtFilename).has_value());
  app->compose();
  app->runAsync();
  app->wait();
}

TEST(TestApp, ValidConfigWithArgumentsAsync) {
  auto app = create_app<PingMultiBroadcastApp>();
  int argc = 2;
  char* argv[] = {const_cast<char*>("TestApp"), const_cast<char*>(kGxeConfigFilename)};

  GXF_ASSERT_TRUE(app->setConfig(argc, argv).has_value());
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_TRUE(app->loadExtensionManifest(kGxeManifestFilename).has_value());
  app->compose();
  GXF_ASSERT_TRUE(app->run().has_value());
}

TEST(TestApp, InvalidConfigWithArguments) {
  auto app = create_app<PingMultiBroadcastApp>();
  int argc = 2;
  char* argv[] = {const_cast<char*>("TestApp"), const_cast<char*>("TestApp")};

  GXF_ASSERT_TRUE(app->setConfig(argc, argv).has_value());
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_FALSE(app->loadExtensionManifest("").has_value());
}

TEST(TestApp, InvalidManifest) {
  auto app = create_app<PingMultiBroadcastApp>();
  int argc = 2;
  char* argv[] = {const_cast<char*>("TestApp"), const_cast<char*>("TestApp")};

  GXF_ASSERT_TRUE(app->setConfig(argc, argv).has_value());
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_FALSE(app->loadExtensionManifest(kInvalidGxFManifestFilename).has_value());
}

TEST(TestApp, EmptyManifest) {
  auto app = create_app<PingMultiBroadcastApp>();
  int argc = 2;
  char* argv[] = {const_cast<char*>("TestApp"), const_cast<char*>("TestApp")};

  GXF_ASSERT_TRUE(app->setConfig(argc, argv).has_value());
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_FALSE(app->loadExtensionManifest(kInvalidGxFManifestFilename).has_value());
}

TEST(TestApp, ValidConfigWithArgumentsWithAbsolutePath) {
  auto app = create_app<PingMultiBroadcastApp>();
  int argc = 2;
  const auto file_path_ = getExecutablePath() + "/" + kGxeConfigFilename;
  char* argv[] = {const_cast<char*>("TestApp"), const_cast<char*>(file_path_.c_str())};

  GXF_ASSERT_TRUE(app->setConfig(argc, argv).has_value());
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_TRUE(app->loadExtensionManifest(kGxeManifestFilename).has_value());
  app->compose();
  GXF_ASSERT_TRUE(app->run().has_value());
}

TEST(TestApp, InvalidConfigWithArgumentsWithAbsolutePath) {
  auto app = create_app<PingMultiBroadcastApp>();
  int argc = 2;
  char* argv[] = {const_cast<char*>("TestApp"),
                  const_cast<char*>("~/Workspace/gxf/registry/config/target_x86_64.yaml")};
  GXF_ASSERT_TRUE(app->setConfig(argc, argv).has_value());
}

}  // namespace gxf
}  // namespace nvidia
