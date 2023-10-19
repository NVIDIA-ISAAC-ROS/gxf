/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/stream/stream_nvscisync.hpp"

#include "gtest/gtest.h"
#include "gxf/core/gxf.h"

namespace nvidia {
namespace gxf {

TEST(StreamSync, CudaCudaSyncSingleStream) {
  constexpr int32_t signaler = static_cast<int32_t>(SyncType::GXF_STREAM_SIGNALER_CUDA);
  constexpr int32_t waiter = static_cast<int32_t>(SyncType::GXF_STREAM_WAITER_CUDA);
  constexpr int32_t signaler_gpu_id = 0;
  constexpr int32_t waiter_gpu_id = 0;

  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::StreamSync", &tid), GXF_SUCCESS);

  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test_stream_sync", &cid), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);

  ASSERT_EQ(GxfParameterSetInt32(context, cid, "signaler", signaler), GXF_SUCCESS);

  ASSERT_EQ(GxfParameterSetInt32(context, cid, "waiter", waiter), GXF_SUCCESS);

  ASSERT_EQ(GxfParameterSetInt32(context, cid, "signaler_gpu_id", signaler_gpu_id), GXF_SUCCESS);

  ASSERT_EQ(GxfParameterSetInt32(context, cid, "waiter_gpu_id", waiter_gpu_id), GXF_SUCCESS);

  Stream* streamSync = static_cast<Stream*>(pointer);
  ASSERT_EQ(streamSync->initialize(), GXF_SUCCESS);

  void *syncObj{nullptr};
  ASSERT_EQ(streamSync->allocate_sync_object(SyncType::GXF_STREAM_SIGNALER_CUDA, SyncType::GXF_STREAM_WAITER_CUDA, reinterpret_cast<void**>(&syncObj)), GXF_SUCCESS);
  ASSERT_NE(syncObj, nullptr);

  cudaStream_t stream{};
  ASSERT_EQ(cudaSetDevice(signaler_gpu_id), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

  ASSERT_EQ(streamSync->setCudaStream(SyncType::GXF_STREAM_SIGNALER_CUDA, stream), GXF_SUCCESS);
  ASSERT_EQ(streamSync->setCudaStream(SyncType::GXF_STREAM_WAITER_CUDA, stream), GXF_SUCCESS);
  // Do some processing on signaler CUDA Stream
  // ....
  // Signal after the processing is submitted
  ASSERT_EQ(streamSync->signalSemaphore(), GXF_SUCCESS);
  // Wait for processing to be finished
  ASSERT_EQ(streamSync->waitSemaphore(), GXF_SUCCESS);

  cudaStreamSynchronize(stream);

  ASSERT_EQ(streamSync->deinitialize(), GXF_SUCCESS);

  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(StreamSync, CudaCudaSyncMultipleStream) {
  constexpr int32_t signaler = static_cast<int32_t>(SyncType::GXF_STREAM_SIGNALER_CUDA);
  constexpr int32_t waiter = static_cast<int32_t>(SyncType::GXF_STREAM_WAITER_CUDA);
  constexpr int32_t signaler_gpu_id = 0;
  constexpr int32_t waiter_gpu_id = 0;

  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {0};
  ASSERT_EQ(GxfCreateEntity(context, &entity_create_info, &eid), GXF_SUCCESS);

  gxf_tid_t tid;
  ASSERT_EQ(GxfComponentTypeId(context, "nvidia::gxf::StreamSync", &tid), GXF_SUCCESS);

  gxf_uid_t cid;
  ASSERT_EQ(GxfComponentAdd(context, eid, tid, "test_stream_sync", &cid), GXF_SUCCESS);

  void* pointer;
  ASSERT_EQ(GxfComponentPointer(context, cid, tid, &pointer), GXF_SUCCESS);

  ASSERT_EQ(GxfParameterSetInt32(context, cid, "signaler", signaler), GXF_SUCCESS);

  ASSERT_EQ(GxfParameterSetInt32(context, cid, "waiter", waiter), GXF_SUCCESS);

  ASSERT_EQ(GxfParameterSetInt32(context, cid, "signaler_gpu_id", signaler_gpu_id), GXF_SUCCESS);

  ASSERT_EQ(GxfParameterSetInt32(context, cid, "waiter_gpu_id", waiter_gpu_id), GXF_SUCCESS);

  Stream* streamSync = static_cast<Stream*>(pointer);
  ASSERT_EQ(streamSync->initialize(), GXF_SUCCESS);

  void *syncObj{nullptr};
  ASSERT_EQ(streamSync->allocate_sync_object(SyncType::GXF_STREAM_SIGNALER_CUDA, SyncType::GXF_STREAM_WAITER_CUDA, reinterpret_cast<void**>(&syncObj)), GXF_SUCCESS);
  ASSERT_NE(syncObj, nullptr);

  cudaStream_t signalerStream{};
  cudaStream_t waiterStream{};
  ASSERT_EQ(cudaSetDevice(signaler_gpu_id), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&signalerStream, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&waiterStream, cudaStreamNonBlocking), cudaSuccess);

  ASSERT_EQ(streamSync->setCudaStream(SyncType::GXF_STREAM_SIGNALER_CUDA, signalerStream), GXF_SUCCESS);
  ASSERT_EQ(streamSync->setCudaStream(SyncType::GXF_STREAM_WAITER_CUDA, waiterStream), GXF_SUCCESS);
  // Do some processing on signaler CUDA Stream
  // ....
  // Signal after the processing is submitted
  ASSERT_EQ(streamSync->signalSemaphore(), GXF_SUCCESS);
  // Wait for processing to be finished
  ASSERT_EQ(streamSync->waitSemaphore(), GXF_SUCCESS);

  ASSERT_EQ(streamSync->deinitialize(), GXF_SUCCESS);

  ASSERT_EQ(cudaStreamDestroy(signalerStream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(waiterStream), cudaSuccess);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

}  // namespace gxf
}  // namespace nvidia
