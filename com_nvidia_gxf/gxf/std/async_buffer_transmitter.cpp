/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/async_buffer_transmitter.hpp"

#include <memory>
#include <utility>

namespace nvidia {
namespace gxf {

gxf_result_t AsyncBufferTransmitter::initialize() {
  size_ = 0;
  return GXF_SUCCESS;
}

gxf_result_t AsyncBufferTransmitter::pop_abi(gxf_uid_t* uid) {
  if (uid == nullptr) { return GXF_ARGUMENT_NULL; }
  if (entity_.is_null()) {
    GXF_LOG_ERROR("Received null entity in double buffer transmitter");
    return GXF_FAILURE;
  }

  // We do not want to decrement the ref count (which will happen in the Entity destructor)
  // as we expect the caller to do that.
  const gxf_result_t code = GxfEntityRefCountInc(context(), entity_.eid());
  if (code != GXF_SUCCESS) { return code; }

  *uid = entity_.eid();
  size_ = 0;
  return GXF_SUCCESS;
}

gxf_result_t AsyncBufferTransmitter::push_abi(gxf_uid_t other) {
  auto maybe = Entity::Shared(context(), other);
  if (!maybe) { return maybe.error(); }

  entity_ = std::move(maybe.value());
  size_ = 1;

  return GXF_SUCCESS;
}

gxf_result_t AsyncBufferTransmitter::peek_abi(gxf_uid_t* uid, int32_t index) {
  // Not supported
  return GXF_FAILURE;
}

size_t AsyncBufferTransmitter::capacity_abi() {
  // to satisfy DownstreamReceptiveSchedulingTerm
  return size_ + 1;
}

size_t AsyncBufferTransmitter::size_abi() {
  return size_;
}

gxf_result_t AsyncBufferTransmitter::publish_abi(gxf_uid_t uid) {
  return push_abi(uid);
}

size_t AsyncBufferTransmitter::back_size_abi() {
  return 0;
}

gxf_result_t AsyncBufferTransmitter::sync_abi() {
  return GXF_SUCCESS;
}

gxf_result_t AsyncBufferTransmitter::sync_io_abi() {
  return GXF_SUCCESS;
}

gxf_result_t AsyncBufferTransmitter::pop_io_abi(gxf_uid_t* uid) {
  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
