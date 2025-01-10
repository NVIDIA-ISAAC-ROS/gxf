/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gtest/gtest.h"

#include "common/assert.hpp"
#include "gxf/app/application.hpp"
#include "gxf/app/entity_group.hpp"
#include "gxf/app/graph_entity.hpp"
#include "gxf/sample/ping_rx.hpp"
#include "gxf/sample/ping_tx.hpp"

namespace nvidia {
namespace gxf {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

class TxResource : public Codelet {
 public:
  virtual ~TxResource() = default;

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->resource(gpu_device_,
      "GPU device resource from which create CUDA streams");
    result &= registrar->resource(thread_resource_, "Thread pool resource");
    result &= registrar->parameter(signal_, "signal", "Signal",
              "Transmitter channel publishing messages to other graph entities");
    result &= registrar->parameter(expected_dev_id_, "expected_dev_id", "Expected dev_id",
              "Expected GPUDevice for testing resource only", -1);
    result &= registrar->parameter(expected_thread_size_, "expected_thread_size",
      "Expected thread_size", "Expected ThreadPool for testing resource only", -1l);
    return ToResultCode(result);
  }

  void test_helper() {
    // get device id from GPUDevice Resource
    if (gpu_device_.try_get()) {
      dev_id_ = gpu_device_.try_get().value()->device_id();
      GXF_LOG_DEBUG("[cid: %ld]: GPUDevice Resource found. Using dev_id: %d",
        cid(), dev_id_);
    } else {
      GXF_LOG_DEBUG("[cid: %ld]: no GPUDevice Resource found. "
        "Using default device id: %d", cid(), dev_id_);
    }

    // get thread pool size from ThreadPool Resource
    if (thread_resource_.try_get()) {
      thread_size_ = thread_resource_.try_get().value()->size();
      GXF_LOG_DEBUG("[cid: %ld]: ThreadPool Resource found. Using size: %ld",
        cid(), thread_size_);
    } else {
      GXF_LOG_DEBUG("[cid: %ld]: no ThreadPool Resource found. "
        "Using default size: %ld", cid(), thread_size_);
    }
  }

  gxf_result_t tick() override {
    test_helper();
    auto message = Entity::New(context());
    if (!message) {
      GXF_LOG_ERROR("Failure creating message entity.");
      return message.error();
    }

    auto result = signal_->publish(message.value());
    GXF_LOG_INFO("Message Sent: %d", this->count);
    this->count = this->count + 1;
    GXF_LOG_INFO("GPUDevice resource: %d, expected: %d; ThreadPool size: %ld, expected: %ld",
      dev_id_, expected_dev_id_.get(), thread_size_, expected_thread_size_.get());
    GXF_ASSERT_EQ(expected_dev_id_, dev_id_);
    GXF_ASSERT_EQ(expected_thread_size_, thread_size_);
    return ToResultCode(message);
  }

 private:
  Parameter<Handle<Transmitter>> signal_;
  int count = 1;
  Resource<Handle<GPUDevice>> gpu_device_;
  int32_t dev_id_ = -1;
  Parameter<int32_t> expected_dev_id_;
  Resource<Handle<ThreadPool>> thread_resource_;
  int64_t thread_size_ = -1;
  Parameter<int64_t> expected_thread_size_;
};

class GPUResourceAppBase : public Application {
 protected:
  void base_compose() {
    // create a codelet to generate 10 messages
    tx_entity = makeEntity<TxResource>("Tx",
                        Arg("expected_dev_id", tx_expected_dev_id),
                        Arg("expected_thread_size", tx_expected_thread_size),
                        makeTerm<CountSchedulingTerm>("count", Arg("count", 10)));

    // create a codelet to receive the messages
    rx_entity = makeEntity<PingRx>("Rx");

    // add data flow connection tx -> rx
    connect(tx_entity, rx_entity);

    // configure the scheduler component
    setScheduler(SchedulerType::kGreedy);
  }
  GraphEntityPtr tx_entity;
  GraphEntityPtr rx_entity;
  int32_t tx_expected_dev_id = -1;
  int32_t rx_expected_dev_id = -1;
  int64_t tx_expected_thread_size = -1;
  int64_t rx_expected_thread_size = -1;
};

class GPUResourceAppDefaultEntityGroup : public GPUResourceAppBase {
 public:
  int32_t DEV_ID = 16;
  int64_t THREAD_SIZE = 8;
  void compose() override {
    this->tx_expected_dev_id = DEV_ID;
    this->tx_expected_thread_size = THREAD_SIZE;
    base_compose();
    // create an entity to contain resources
    auto src_entity = makeEntity("resource_entity",
      makeResource<GPUDevice>("test_device", Arg("dev_id", DEV_ID)),
      makeResource<ThreadPool>("test_pool", Arg("initial_size", THREAD_SIZE)));
  }
};

class GPUResourceAppUserEntityGroup : public GPUResourceAppBase {
 public:
  int32_t DEV_ID_0 = 16;
  int64_t THREAD_SIZE_0 = 15;
  int32_t DEV_ID_1 = 32;
  int64_t THREAD_SIZE_1 = 31;
  void compose() override {
    this->tx_expected_dev_id = DEV_ID_1;
    this->tx_expected_thread_size = THREAD_SIZE_1;
    base_compose();
    // create an entity to contain resources
    auto src_entity_0 = makeEntity("resource_entity_0",
      makeResource<GPUDevice>("test_device_0", Arg("dev_id", DEV_ID_0)),
      makeResource<ThreadPool>("test_pool_0", Arg("initial_size", THREAD_SIZE_0)));
    // create an entity to contain resources
    auto src_entity_1 = makeEntity("resource_entity_1",
      makeResource<GPUDevice>("test_device_1", Arg("dev_id", DEV_ID_1)),
      makeResource<ThreadPool>("test_pool_1", Arg("initial_size", THREAD_SIZE_1)));
    auto eg_0 = makeEntityGroup("entity_group_0");
    eg_0->add(tx_entity);
    eg_0->add(src_entity_1);
  }
};

class GPUResourceAppUserEntityGroup2 : public GPUResourceAppBase {
 public:
  int32_t DEV_ID_0 = 16;
  int64_t THREAD_SIZE_0 = 15;
  void compose() override {
    this->tx_expected_dev_id = -1;
    this->tx_expected_thread_size = -1;
    base_compose();
    // create an entity to contain resources, appliable to default entity group
    auto src_entity_0 = makeEntity("resource_entity_0",
      makeResource<GPUDevice>("test_device_0", Arg("dev_id", DEV_ID_0)),
      makeResource<ThreadPool>("test_pool_0", Arg("initial_size", THREAD_SIZE_0)));
    // create a user entity group, without adding any resource
    auto eg_0 = makeEntityGroup("entity_group_0");
    eg_0->add(tx_entity);
  }
};

TEST(TestApp, GPUResourceAppDefaultEntityGroup) {
  auto app = create_app<GPUResourceAppDefaultEntityGroup>();
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_TRUE(app->loadExtensionManifest(kGxeManifestFilename).has_value());
  app->compose();
  GXF_ASSERT_TRUE(app->run().has_value());
}

TEST(TestApp, GPUResourceAppUserEntityGroup) {
  auto app = create_app<GPUResourceAppUserEntityGroup>();
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_TRUE(app->loadExtensionManifest(kGxeManifestFilename).has_value());
  app->compose();
  GXF_ASSERT_TRUE(app->run().has_value());
}

TEST(TestApp, GPUResourceAppUserEntityGroup2) {
  auto app = create_app<GPUResourceAppUserEntityGroup2>();
  GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
  GXF_ASSERT_TRUE(app->loadExtensionManifest(kGxeManifestFilename).has_value());
  app->compose();
  GXF_ASSERT_TRUE(app->run().has_value());
}

}  // namespace gxf
}  // namespace nvidia
