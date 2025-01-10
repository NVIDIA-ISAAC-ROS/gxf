/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/cuda/cuda_common.hpp"
#include "gxf/cuda/cuda_stream.hpp"

#include <queue>
#include <string>
#include <utility>

namespace nvidia {
namespace gxf {

CudaStream::~CudaStream() {
  deinitialize();
}

Expected<void> CudaStream::initialize(uint32_t flags, int dev_id, int32_t priority) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  if (stream_) {
    GXF_LOG_ERROR("cuda stream is already available, failed to initialize");
    return Unexpected{GXF_FAILURE};
  }
  if (dev_id >= 0) {
    cudaError_t result = cudaSetDevice(dev_id);
    CHECK_CUDA_ERROR(result, "Failure setting device id: %d to create cudastream", dev_id);
  }
  dev_id_ = dev_id;
  cudaError_t result = cudaStreamCreateWithPriority(&stream_, flags, priority);
  CHECK_CUDA_ERROR(result, "Failure creating CUDA stream");

  return prepareResourceInternal(dev_id);
}

Expected<void> CudaStream::prepareResourceInternal(int dev_id) {
  auto event = CudaEvent::createEventInternal(cudaEventDefault, dev_id);
  if (!event) {
    GXF_LOG_ERROR("Failure creating CudaStream's sync_event.");
    return ForwardError(event);
  }
  sync_event_ = std::move(event.value());
  GXF_ASSERT(sync_event_, "sync_event_ cannot be null");

  return Success;
}

Expected<void> CudaStream::deinitialize() {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  if (stream_) {
    if (dev_id_ >= 0) {
      CHECK_CUDA_ERROR(cudaSetDevice(dev_id_),
          "Failure setting device id: %d to destroy cudastream", dev_id_);
    }
    cudaError_t result = cudaStreamDestroy(stream_);
    CHECK_CUDA_ERROR(result, "Failure destroying CUDA stream");
    GXF_LOG_DEBUG("CudaStream destroyed");
  }
  resetEventsInternal(recorded_event_queue_);
  sync_event_.reset();
  stream_ = 0;
  dev_id_ = 0;
  return Success;
}

Expected<cudaStream_t> CudaStream::stream() const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  if (0 == stream_) {
    GXF_LOG_ERROR("CUDA stream not provided");
    return Unexpected{GXF_FAILURE};
  }
  return stream_;
}

Expected<void> CudaStream::recordEventInternal(cudaEvent_t e) {
  GXF_ASSERT(e, "event is null");
  CHECK_CUDA_ERROR(cudaEventRecord(e, stream_), "Failure recording cuda event on stream");
  return Success;
}

Expected<void> CudaStream::syncEventInternal(cudaEvent_t e) {
  GXF_ASSERT(e, "event is null");
  CHECK_CUDA_ERROR(cudaEventSynchronize(e), "Failure syncing cuda event on stream");
  return Success;
}

Expected<void> CudaStream::record(
  Handle<CudaEvent> event, Entity input_entity, SyncedCallback synced_cb) {
  if (event.is_null()) {
    return Unexpected{GXF_ARGUMENT_NULL};
  }
  auto event_id = event->event();
  if (!event_id) {
    return ForwardError(event_id);
  }

  gxf_uid_t eid = kNullUid;
  auto ret = GxfComponentEntity(event.context(), event.cid(), &eid);
  if (ret != GXF_SUCCESS || eid == kNullUid) {
    GXF_LOG_ERROR("Failure creating stream event from CudaEvent handle, event entity not found");
    return Unexpected{GXF_ARGUMENT_NULL};
  }
  // Clone event's entity
  auto event_entity = Entity::Shared(event.context(), eid);
  if (!event_entity) {
    return ForwardError(event_entity);
  }

  // keep event_entity, input_entity and callback synced_cb once event synced.
  return record(event->event().value(),
    [e = std::move(event_entity.value()), in = std::move(input_entity),
     synced_cb = std::move(synced_cb)](cudaEvent_t) {
      if (synced_cb) {
        synced_cb();
      }
    });
}

Expected<void> CudaStream::record(cudaEvent_t event, CudaStream::EventDestroy cb) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  auto ret = recordEventInternal(event);
  if (!ret) {
    GXF_LOG_ERROR("Failure recording event and callback on cudastream");
    return ForwardError(ret);
  }
  auto event_ptr = CudaEvent::createEventInternal(event, cb);
  if (!event_ptr) {
    GXF_LOG_ERROR("Failure recording event since wrap stream event failed.");
    return ForwardError(event_ptr);
  }
  GXF_ASSERT(event_ptr.value() && *(event_ptr.value()), "event_ptr is empty");
  GXF_LOG_DEBUG("Successfully recording a event");
  recorded_event_queue_.emplace(std::move(event_ptr.value()));
  return Success;
}

Expected<void> CudaStream::resetEvents() {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  GXF_LOG_DEBUG("Reset all events in Cuda Stream");
  return resetEventsInternal(recorded_event_queue_);
}

Expected<void> CudaStream::resetEventsInternal(std::queue<CudaEvent::EventPtr>& q) {
  while (!q.empty()) {
      q.pop();
  }
  return Success;
}

Expected<void> CudaStream::syncStream() {
  std::queue<CudaEvent::EventPtr> pendings;
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  pendings.swap(recorded_event_queue_);
  GXF_ASSERT(sync_event_, "sync_event_ is empty");
  auto ret = recordEventInternal(*sync_event_);
  lock.unlock();

  while (!pendings.empty()) {
    CudaEvent::EventPtr event = std::move(pendings.front());
    GXF_ASSERT(event, "pending event is null");
    pendings.pop();
    ret &= syncEventInternal(*event);
    event.reset();
  }

  lock.lock();
  ret &= syncEventInternal(*sync_event_);
  lock.unlock();

  if (!ret) {
    GXF_LOG_ERROR("Failure syncing on cudastream");
    return ForwardError(ret);
  }

  GXF_LOG_DEBUG("Successfully syncing on cudastream");
  return Success;
}

}  // namespace gxf
}  // namespace nvidia
