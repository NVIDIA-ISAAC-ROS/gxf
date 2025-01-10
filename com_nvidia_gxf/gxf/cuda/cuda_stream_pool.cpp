/*
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/cuda/cuda_stream_pool.hpp"

#include <memory>
#include <utility>

#include "gxf/core/common_expected_macro.hpp"
#include "gxf/cuda/cuda_common.hpp"
#include "gxf/cuda/cuda_stream.hpp"

namespace nvidia {
namespace gxf {

namespace {
constexpr uint32_t kDefaultStreamFlags = cudaStreamNonBlocking;
constexpr int32_t kDefaultStreamPriority = 0;
constexpr uint32_t kDefaultReservedSize = 1;
constexpr uint32_t kDefaultMaxSize = 0;
constexpr int32_t kDefaultDeviceId = 0;
constexpr const char* kDefaultStreamName = "CudaStream";
}

CudaStreamPool::~CudaStreamPool() {}

gxf_result_t CudaStreamPool::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->resource(gpu_device_, "GPU device resource from which create CUDA streams");
  result &= registrar->parameter(
      stream_flags_, "stream_flags", "Stream Flags",
      "Create CUDA streams with flags.", kDefaultStreamFlags);
  result &= registrar->parameter(
      stream_priority_, "stream_priority", "Stream Priority",
      "Create CUDA streams with priority.", kDefaultStreamPriority);
  result &= registrar->parameter(
      reserved_size_, "reserved_size", "Reserved Stream Size",
      "Reserve several CUDA streams before 1st request coming", kDefaultReservedSize);
  result &= registrar->parameter(
      max_size_, "max_size", "Maximum Stream Size",
      "The maximum stream size for the pool to allocate, unlimited by default", kDefaultMaxSize);
  return ToResultCode(result);
}

gxf_result_t CudaStreamPool::initialize() {
  // get device id from GPUDevice Resource
  if (gpu_device_.try_get()) {
    dev_id_ = gpu_device_.try_get().value()->device_id();
    GXF_LOG_DEBUG("CudaStreamPool [cid: %ld]: GPUDevice Resource found. Using dev_id: %d",
      cid(), dev_id_);
  } else {
    dev_id_ = kDefaultDeviceId;
    GXF_LOG_DEBUG("CudaStreamPool [cid: %ld]: no GPUDevice Resource found. "
      "Using default device id: %d", cid(), dev_id_);
  }

  std::unique_lock<std::mutex> lock(mutex_);
  uint32_t reserve_size = reserved_size_.get();
  if (max_size_.get() && max_size_.get() < reserve_size) {
    GXF_LOG_WARNING("stream pool max_size: %u < reserved_size: %u, reset max_size",
                     max_size_.get(), reserve_size);
    auto ret = max_size_.set(reserve_size);
    if (!ret) {
      GXF_LOG_ERROR("stream pool reset max_size to %u failed.", reserve_size);
      return ToResultCode(ret);
    }
  }
  auto result = reserveStreams();
  stage_ = AllocatorStage::kInitialized;

  return ToResultCode(result);
}

gxf_result_t CudaStreamPool::deinitialize() {
  std::unique_lock<std::mutex> lock(mutex_);
  streams_.clear();
  reserved_streams_ = {};
  stage_ = AllocatorStage::kUninitialized;
  return GXF_SUCCESS;
}

gxf_result_t CudaStreamPool::is_available_abi(uint64_t size) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (stage_ != AllocatorStage::kInitialized) {
    GXF_LOG_ERROR("Allocator must be in Initialized stage before starting."
                  " Current state is %s", allocator_stage_str(stage_));
    return GXF_INVALID_LIFECYCLE_STAGE;
  }
  if (!max_size_.get()) {
    return GXF_SUCCESS;
  }
  if (streams_.size() + size < (size_t)max_size_.get()) {
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

// Allocate stream entity
gxf_result_t CudaStreamPool::allocate_abi(uint64_t size, int32_t type, void** pointer) {
  if (stage_ != AllocatorStage::kInitialized) {
    GXF_LOG_ERROR("Allocator must be in Initialized stage before starting."
                  " Current state is %s", allocator_stage_str(stage_));
    return GXF_INVALID_LIFECYCLE_STAGE;
  }
  if (size != 1LL) {
    GXF_LOG_ERROR("CudaStreamPool does not support multiple cudaStream allocation "
                  "in a single call, size must be 1");
    return GXF_ARGUMENT_INVALID;
  }
  std::unique_lock<std::mutex> lock(mutex_);
  if (max_size_.get() && streams_.size() >= (size_t)max_size_.get()) {
    GXF_LOG_ERROR("CudaStreamPool reached capacity(%u), could not allocate more streams",
                  max_size_.get());
    return GXF_OUT_OF_MEMORY;
  }

  Expected<Entity> stream_entity = Unexpected{GXF_ENTITY_NOT_FOUND};
  if (reserved_streams_.empty()) {
    stream_entity = createNewStreamEntity();
    GXF_LOG_DEBUG("Allocate new cudastream");
  } else {
    stream_entity = std::move(reserved_streams_.front());
    reserved_streams_.pop();
    GXF_LOG_DEBUG("Allocate cudastream from reserved streams");
  }

  if (!stream_entity) {
    GXF_LOG_DEBUG("Allocated stream entity is null");
    return ToResultCode(stream_entity);
  }

  gxf_uid_t eid = stream_entity.value().eid();
  // make entity available for returned pointer
  auto entity = std::make_unique<Entity>(std::move(stream_entity.value()));
  GXF_ASSERT(entity, "entity ptr is null");
  GXF_LOG_DEBUG("Allocated cuda stream successfully");
  // return Entity pointer
  *pointer = reinterpret_cast<void*>(entity.get());
  streams_.emplace(eid, std::move(entity));
  return GXF_SUCCESS;
}

gxf_result_t CudaStreamPool::free_abi(void* pointer) {
  Entity* entity = reinterpret_cast<Entity*>(pointer);
  GXF_ASSERT(entity, "free_abi pointer is null");
  GXF_LOG_DEBUG("Freeing cuda stream");
  std::unique_lock<std::mutex> lock(mutex_);
  auto iter = streams_.find(entity->eid());
  if (iter == streams_.end()) {
    GXF_LOG_ERROR("Failed to find cuda stream eid: %05zu in allocated streams.", entity->eid());
    return GXF_ARGUMENT_OUT_OF_RANGE;
  }
  Entity stream_entity = std::move(*(iter->second));
  streams_.erase(iter);
  auto stream = stream_entity.get<CudaStream>();
  if (!stream) {
    GXF_LOG_ERROR("free_abi received wrong entity which doesn't have cudastream");
    return GXF_ARGUMENT_OUT_OF_RANGE;
  }
  auto ret = stream.value()->resetEvents();
  if (!ret) {
    GXF_LOG_ERROR("Failed to free cuda stream entity due to resetting stream events failed.");
    return ToResultCode(ret);
  }
  reserved_streams_.emplace(std::move(stream_entity));

  return GXF_SUCCESS;
}

Expected<Handle<CudaStream>> CudaStreamPool::allocateStream() {
  if (stage_ != AllocatorStage::kInitialized) {
    GXF_LOG_ERROR("Allocator must be in Initialized stage before starting."
                  " Current state is %s", allocator_stage_str(stage_));
    return Unexpected{GXF_INVALID_LIFECYCLE_STAGE};
  }
  // allocate stream entity pointer
  const auto maybe = allocate(1, MemoryStorageType::kDevice);
  if (!maybe) {
    GXF_LOG_ERROR("allocate cudastream failed.");
    return ForwardError(maybe);
  }
  Entity* stream_entity = ValuePointer<Entity>(maybe.value());
  GXF_ASSERT(stream_entity, "stream_entity pointer is null");
  auto stream = stream_entity->get<CudaStream>(kDefaultStreamName);
  GXF_ASSERT(stream, "get stream:%s failed in allocation", kDefaultStreamName);
  return stream;
}

Expected<void> CudaStreamPool::releaseStream(Handle<CudaStream> stream) {
  if (stream.is_null()) {
    GXF_LOG_ERROR("releaseStream must have valid stream parameters");
    return Unexpected{GXF_ARGUMENT_INVALID};
  }

  GXF_ASSERT(stream.context() == context(), "cudastream context doesn't match pool's context");
  gxf_uid_t eid = kNullUid;
  auto ret = GxfComponentEntity(stream.context(), stream.cid(), &eid);
  if (ret != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to find CudaStream(cid: %zu) entity to release", stream.cid());
    return ExpectedOrCode(ret);
  }
  auto stream_entity = Entity::Shared(stream.context(), eid);
  if (!stream_entity) {
    return ForwardError(stream_entity);
  }
  return this->free(BytePointer(&stream_entity.value()));
}

Expected<Entity> CudaStreamPool::createNewStreamEntity() {
  auto owner = Entity::New(context());
  if (!owner) {
    return ForwardError(owner);
  }

  auto stream = owner.value().add<CudaStream>(kDefaultStreamName);
  if (!stream) {
    return ForwardError(stream);
  }
  auto result = stream.value()->initialize(stream_flags_, dev_id_, stream_priority_);
  if (!result) {
    GXF_LOG_ERROR("create new cuda stream failed during initialization");
    return ForwardError(result);
  }
  return owner;
}

Expected<void> CudaStreamPool::reserveStreams() {
  GXF_ASSERT(reserved_streams_.empty(), "reserved_streams_ should be empty before reserve");
  for (uint32_t i = 0; i < reserved_size_.get(); ++i) {
    auto stream = createNewStreamEntity();
    if (!stream) {
      return ForwardError(stream);
    }
    reserved_streams_.emplace(std::move(stream.value()));
  }
  GXF_ASSERT_EQ(reserved_streams_.size(), (size_t)reserved_size_.get());
  return Success;
}

}  // namespace gxf
}  // namespace nvidia
