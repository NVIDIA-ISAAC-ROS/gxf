/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/async_buffer_receiver.hpp"

#include <memory>
#include <utility>

namespace nvidia {
namespace gxf {

void AsyncBufferReceiver::reset_buffer() {
  freshest_ = 0;
  reading_ = 0;
  slots_[0] = 0;
  slots_[1] = 0;
  is_filled_first_time_ = false;
}

gxf_result_t AsyncBufferReceiver::initialize() {
  this->reset_buffer();
  return GXF_SUCCESS;
}

gxf_result_t AsyncBufferReceiver::deinitialize() {
  this->reset_buffer();
  return GXF_SUCCESS;
}

Entity AsyncBufferReceiver::read_freshest() {
  // Simpson's protocol reading algorithm
  int reading_pair = freshest_;
  reading_ = reading_pair;
  int reading_slot = slots_[reading_pair];
  return entity_data_[reading_pair][reading_slot];
}

gxf_result_t AsyncBufferReceiver::pop_abi(gxf_uid_t* uid) {
  if (uid == nullptr) { return GXF_ARGUMENT_NULL; }

  Entity entity_;
  if (is_filled_first_time_) {
    entity_ = read_freshest();
    // We do not want to decrement the ref count (which will happen in the Entity destructor)
    // as we expect the caller to do that.
    const gxf_result_t code = GxfEntityRefCountInc(context(), entity_.eid());
    if (code != GXF_SUCCESS) { return code; }
  }  // else null entity will be passed, and the receiver needs to handle it

  *uid = entity_.eid();
  return GXF_SUCCESS;
}

gxf_result_t AsyncBufferReceiver::push_abi(gxf_uid_t other) {
  auto maybe = Entity::Shared(context(), other);
  if (!maybe) { return maybe.error(); }

  // Simpson's protocol writing algorithm
  int writing_pair = 1 - reading_;
  int writing_slot = 1 - slots_[writing_pair];
  entity_data_[writing_pair][writing_slot] = std::move(maybe.value());
  slots_[writing_pair] = writing_slot;
  freshest_ = writing_pair;
  if (!is_filled_first_time_) { is_filled_first_time_ = true; }

  return GXF_SUCCESS;
}

gxf_result_t AsyncBufferReceiver::peek_abi(gxf_uid_t* uid, int32_t index) {
  // Not supported
  return GXF_FAILURE;
}

gxf_result_t AsyncBufferReceiver::peek_back_abi(gxf_uid_t* uid, int32_t index) {
  // Not supported
  return GXF_FAILURE;
}

size_t AsyncBufferReceiver::capacity_abi() {
  // to satisfy MessageAvailableSchedulingTerm
  return size_abi() + 1;
}

size_t AsyncBufferReceiver::size_abi() {
  return 1;
}

gxf_result_t AsyncBufferReceiver::receive_abi(gxf_uid_t* uid) {
  return pop_abi(uid);
}

size_t AsyncBufferReceiver::back_size_abi() {
  return 0;
}

gxf_result_t AsyncBufferReceiver::sync_abi() {
  return GXF_SUCCESS;
}

gxf_result_t AsyncBufferReceiver::sync_io_abi() {
  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
