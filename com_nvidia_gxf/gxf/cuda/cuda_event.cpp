/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/cuda/cuda_common.hpp"
#include "gxf/cuda/cuda_event.hpp"

#include <utility>

namespace nvidia {
namespace gxf {

Expected<CudaEvent::EventPtr> CudaEvent::createEventInternal(uint32_t flags, int gpuid) {
  cudaError_t result = cudaSuccess;
  if (gpuid >= 0) {
    result = cudaSetDevice(gpuid);
    CHECK_CUDA_ERROR(result, "Failure setting device id: %d to create cudaevent", gpuid);
  }
  cudaEvent_t cu_event = 0;
  result = cudaEventCreateWithFlags(&cu_event, flags);
  CHECK_CUDA_ERROR(result, "Failure creating internal event");
  GXF_ASSERT(cu_event, "event null");
  return createEventInternal(cu_event, [gpuid](cudaEvent_t e){
    if (gpuid >= 0) {
      CONTINUE_CUDA_ERROR(cudaSetDevice(gpuid),
                       "Failure setting device id: %d to create cudaevent", gpuid);
    }
    CONTINUE_CUDA_ERROR(cudaEventDestroy(e), "Failure destroying internal event");
  });
}

Expected<CudaEvent::EventPtr> CudaEvent::createEventInternal(
    cudaEvent_t event, EventDestroy free_event) {
  GXF_ASSERT(event, "event null");
  CudaEvent::EventPtr ret(new cudaEvent_t(event),
          [free_event = std::move(free_event)](cudaEvent_t* e){
            if (*e && free_event) { free_event(*e); }
            delete e;
          });
  if (!ret) {
    GXF_LOG_ERROR("New EventPtr failed.");
    return Unexpected{GXF_OUT_OF_MEMORY};
  }
  return ret;
}

CudaEvent::~CudaEvent() {
  resetInternal();
}

Expected<void> CudaEvent::initWithEvent(cudaEvent_t event, int dev_id, EventDestroy free_fnc) {
  if (!event) {
      GXF_LOG_ERROR("init with empty event");
      return Unexpected{GXF_ARGUMENT_INVALID};
  }
  if (event_) {
    GXF_LOG_DEBUG("event pointer already exist, re-init with new event");
    resetInternal();
  }
  GXF_ASSERT(!event_, "Internal event must be null");
  auto new_event = createEventInternal(event, free_fnc);
  if (!new_event) {
      GXF_LOG_DEBUG("Failed to create new cuda event");
      return Unexpected{GXF_FAILURE};
  }
  dev_id_ = dev_id;
  event_ = std::move(new_event.value());
  GXF_ASSERT(event_ && *event_, "inited event is invalid");
  return Success;
}

Expected<void> CudaEvent::init(uint32_t flags, int dev_id) {
  if (event_) {
    GXF_LOG_DEBUG("event pointer already exist, re-init to new event");
    resetInternal();
  }
  GXF_ASSERT(!event_, "Internal event must be null");
  auto new_event = createEventInternal(flags, dev_id);
  if (!new_event) {
      GXF_LOG_DEBUG("Failed to create new cuda event");
      return Unexpected{GXF_FAILURE};
  }
  dev_id_ = dev_id;
  event_ = std::move(new_event.value());
  GXF_ASSERT(event_ && *event_, "inited event is invalid");
  return Success;
}

Expected<void> CudaEvent::deinit() {
    resetInternal();
    return Success;
}

void CudaEvent::resetInternal() {
  if (event_) {
    event_.reset();
    event_ = EventPtr(nullptr, [](cudaEvent_t*){});
    dev_id_ = -1;
  }
}

Expected<cudaEvent_t> CudaEvent::event() const {
  if (!event_) {
      return Unexpected{GXF_FAILURE};
  }
  return *event_;
}

}  // namespace gxf
}  // namespace nvidia
