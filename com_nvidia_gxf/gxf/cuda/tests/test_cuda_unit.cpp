/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <chrono>
#include <thread>

#include "gtest/gtest.h"

#include "common/assert.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/handle.hpp"
#include "gxf/std/double_buffer_receiver.hpp"

#include "gxf/cuda/cuda_stream_id.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/cuda/cuda_event.hpp"

using namespace nvidia::gxf;

namespace {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";
constexpr const char* kGxfCudaUnitTestFilename = "gxf/cuda/tests/test_cuda_unit.yaml";

}  // namespace


TEST(CudaUnit, CudaStreamEvent) {
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  // GXF_ASSERT_SUCCESS(GxfSetSeverity(context, static_cast<gxf_severity_t>(4)));
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info, &eid));

  gxf_tid_t tid;
  GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::CudaStreamPool", &tid));
  gxf_uid_t cid;
  GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid, tid, "cudastreampool", &cid));
  auto maybe_pool = Handle<CudaStreamPool>::Create(context, cid);
  ASSERT_TRUE(maybe_pool && maybe_pool.value());
  auto & pool = maybe_pool.value();
  GXF_ASSERT_SUCCESS(pool->initialize());
  auto maybe_stream = pool->allocateStream();
  ASSERT_TRUE(maybe_stream);
  ASSERT_TRUE(maybe_stream.value());
  auto & stream = maybe_stream.value();
  ASSERT_TRUE(stream->stream());
  ASSERT_TRUE(stream->dev_id() == 0);

  cudaEvent_t event_id;
  ASSERT_EQ(cudaEventCreateWithFlags(&event_id, 0), cudaSuccess);
  CudaEvent event;
  ASSERT_TRUE(event.initWithEvent(event_id, stream->dev_id(), [](cudaEvent_t e){ cudaEventDestroy(e); }));
  ASSERT_TRUE(event.event());

  ASSERT_TRUE(stream->record(event.event().value(), [&event](cudaEvent_t){ event.deinit(); }));
  ASSERT_TRUE(stream->syncStream());
  // event destroyed after syncStream
  ASSERT_FALSE(event.event());

  GxfEntityDestroy(context, eid);
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

class TestCudaStream : public testing::Test {
 public:
  void SetUp() override { }
  void TearDown() override { deinitContext(); }

  void initContext() {
    gxf_context_t context;
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));

    const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
    GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context, kGxfCudaUnitTestFilename));
    GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
    GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
    context_ = context;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  void stopContext() {
    if (!context_) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    GXF_ASSERT_SUCCESS(GxfGraphInterrupt(context_));
    GXF_ASSERT_SUCCESS(GxfGraphWait(context_));
  }

  void deinitContext() {
    if (!context_) {
      return;
    }
    GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context_));
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context_));
    context_ = kNullContext;
  }

  const char* kGPUDeviceType = nvidia::TypenameAsString<nvidia::gxf::GPUDevice>();
  void findStreamPool() {
    gxf_uid_t eid;
    GXF_ASSERT_SUCCESS(GxfEntityFind(context_, "cuda_stream_pool", &eid));
    gxf_tid_t tid;
    GXF_ASSERT_SUCCESS(
        GxfComponentTypeId(context_, nvidia::TypenameAsString<CudaStreamPool>(), &tid));
    gxf_uid_t cid;
    GXF_ASSERT_SUCCESS(
        GxfComponentFind(context_, eid, tid, "stream_pool", nullptr, &cid));
    auto maybe_pool = Handle<CudaStreamPool>::Create(context_, cid);
    ASSERT_TRUE(maybe_pool) << "CudastreamPool is not found";
    Handle<CudaStreamPool>& pool = maybe_pool.value();
    ASSERT_TRUE(pool);
    stream_pool_ = pool;

    // get dev_id_
    gxf_uid_t cid_gpu_device;
    GXF_ASSERT_SUCCESS(GxfEntityResourceGetHandle(context_, eid, kGPUDeviceType, "GPU_0", &cid_gpu_device));
    auto maybe_gpu_device = nvidia::gxf::Handle<nvidia::gxf::GPUDevice>::Create(context_, cid_gpu_device);
    ASSERT_TRUE(maybe_gpu_device) << "GPUDevice resource is not found";
    dev_id_ = maybe_gpu_device.value()->device_id();
  }

  void findSyncReceiver() {
    gxf_uid_t sync_eid;
    GXF_ASSERT_SUCCESS(GxfEntityFind(context_, "sync", &sync_eid));

    gxf_tid_t rx_tid;
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context_, "nvidia::gxf::DoubleBufferReceiver", &rx_tid));

    int offset = 0;
    gxf_uid_t rx_cid;
    GXF_ASSERT_SUCCESS(GxfComponentFind(context_, sync_eid, rx_tid, "rx1", &offset, &rx_cid));
    auto rx = Handle<DoubleBufferReceiver>::Create(context_, rx_cid);
    ASSERT_TRUE(rx);
    ASSERT_TRUE(rx.has_value());
    sync_rx_ = rx.value();
  }

  gxf_context_t getContext() const { return context_; }
  Handle<CudaStreamPool> getStreamPool() const { return stream_pool_; }
  Handle<DoubleBufferReceiver> getSyncReceiver() const { return sync_rx_; }
  int32_t getDevId() const { return dev_id_; }

 protected:
  gxf_context_t context_;
  Handle<CudaStreamPool> stream_pool_;
  Handle<DoubleBufferReceiver> sync_rx_;
  int32_t dev_id_ = -1;
};

TEST_F(TestCudaStream, TestSingleStreamEvent) {
  ASSERT_NO_FATAL_FAILURE(initContext());
  gxf_context_t context = getContext();
  ASSERT_TRUE(context);
  ASSERT_NO_FATAL_FAILURE(findStreamPool());
  auto pool = getStreamPool();
  ASSERT_TRUE(pool) << "CudastreamPool handle is null";

  ASSERT_NO_FATAL_FAILURE(findSyncReceiver());
  auto rx = getSyncReceiver();
  ASSERT_TRUE(pool) << "CudastreamSync receiver is null";

  auto maybe_stream = pool->allocateStream();
  ASSERT_TRUE(maybe_stream) << "allocate Cuda Stream failed";
  Handle<CudaStream>& stream = maybe_stream.value();
  ASSERT_TRUE(stream) << "Cudastream handle is null";

  auto maybe_stream_t = stream->stream();
  ASSERT_TRUE(maybe_stream_t);
  ASSERT_TRUE(maybe_stream_t.value() != 0);

  for (int i = 0; i < 10; ++i) {
    auto message = Entity::New(context);
    ASSERT_TRUE(message);
    auto maybe_event = message.value().add<CudaEvent>("event");
    ASSERT_TRUE(maybe_event);
    auto maybe_stream_id = message.value().add<CudaStreamId>("stream0");
    ASSERT_TRUE(maybe_stream_id);
    maybe_stream_id.value()->stream_cid = stream.cid();
    Handle<CudaEvent>& event = maybe_event.value();
    ASSERT_TRUE(event);
    ASSERT_TRUE(event->init(0, getDevId()));
    ASSERT_TRUE(stream->record(event, Entity())) << "record event failed.";
    rx->push(std::move(message.value()));
    rx->sync();
  }

  ASSERT_NO_FATAL_FAILURE(stopContext());

  ASSERT_TRUE(stream->syncStream()) << "record event failed.";
  ASSERT_TRUE(pool->releaseStream(stream)) << "release stream failed";
}

TEST_F(TestCudaStream, TestMultipleStreamEvent) {
  ASSERT_NO_FATAL_FAILURE(initContext());
  gxf_context_t context = getContext();
  ASSERT_TRUE(context);
  ASSERT_NO_FATAL_FAILURE(findStreamPool());
  auto pool = getStreamPool();
  ASSERT_TRUE(pool) << "CudastreamPool handle is null";

  ASSERT_NO_FATAL_FAILURE(findSyncReceiver());
  auto rx = getSyncReceiver();
  ASSERT_TRUE(pool) << "CudastreamSync receiver is null";

  std::vector<Handle<CudaStream>> streams;
  for (int i = 0; i < 5; ++i) {
    auto maybe_stream = pool->allocateStream();
    ASSERT_TRUE(maybe_stream) << "allocate Cuda Stream " << i << " failed";
    Handle<CudaStream>& stream = maybe_stream.value();
    ASSERT_TRUE(stream) << "Cudastream handle is null";
    streams.push_back(stream);
    auto maybe_stream_t = stream->stream();
    ASSERT_TRUE(maybe_stream_t);
    ASSERT_TRUE(maybe_stream_t.value() != 0);

    auto message = Entity::New(context);
    ASSERT_TRUE(message);
    auto maybe_event = message.value().add<CudaEvent>("event");
    ASSERT_TRUE(maybe_event);
    auto maybe_stream_id = message.value().add<CudaStreamId>("stream");
    ASSERT_TRUE(maybe_stream_id);
    maybe_stream_id.value()->stream_cid = stream.cid();
    Handle<CudaEvent>& event = maybe_event.value();
    ASSERT_TRUE(event);
    ASSERT_TRUE(event->init(0, getDevId()));
    ASSERT_TRUE(stream->record(event, Entity())) << "record event failed.";
    rx->push(std::move(message.value()));
    rx->sync();
  }

  ASSERT_NO_FATAL_FAILURE(stopContext());
}
