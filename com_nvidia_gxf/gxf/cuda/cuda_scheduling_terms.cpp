/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <string>
#include <utility>

#include "gxf/cuda/cuda_buffer.hpp"
#include "gxf/cuda/cuda_common.hpp"
#include "gxf/cuda/cuda_event.hpp"
#include "gxf/cuda/cuda_scheduling_terms.hpp"
#include "gxf/cuda/cuda_stream.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t CudaStreamSchedulingTerm::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      receiver_, "receiver", "Receiver queue",
      "The receiver queue on which the scheduling term checks for the data"
      "readiness on cuda stream");
  return ToResultCode(result);
}

void CUDART_CB CudaStreamSchedulingTerm::cudaHostCallback(void* term_ptr) {
  auto term = reinterpret_cast<CudaStreamSchedulingTerm*>(term_ptr);
  GXF_LOG_VERBOSE("Received host callback from cuda stream for entity [E%05" PRId64 "]",
                  term->receiver_->eid());

  auto cb_registered = State::CALLBACK_REGISTERED;
  GXF_ASSERT_TRUE(term->state_.compare_exchange_strong(cb_registered, State::DATA_AVAILABLE));
  GxfEntityEventNotify(term->receiver_->context(), term->receiver_->eid());
}

gxf_result_t CudaStreamSchedulingTerm::update_state_abi(int64_t timestamp) {
  Expected<Entity> message = Unexpected{GXF_FAILURE};
  auto current_state = state_.load();
  // cuda stream wait is over, message can be dequeued
  if (current_state == State::DATA_AVAILABLE || current_state == State::CALLBACK_REGISTERED) {
    return GXF_SUCCESS;
  }

  // scheduling term is currently not monitoring any message, check for new incoming messages
  if (current_state == State::UNSET) {
    // Sync any messages from the backstage before peeking into the messages
    if (receiver_->size() == 0) {
      auto code = receiver_->sync();
      if (!code) { return ToResultCode(code); }
    }

    // no new messages, no change in state
    if (receiver_->size() == 0) { return GXF_SUCCESS; }

    auto message = receiver_->peek(0);
    if (!message || message.value().is_null()) {
      GXF_LOG_ERROR("Received invalid message at receiver [C%05ld]", receiver_->cid());
      return GXF_FAILURE;
    }

    message_eid_ = message.value().eid();
    auto stream_id = message->get<CudaStreamId>();
    // message without a cuda stream component, no action taken
    // message must be consumed before we can process the next one in the queue
    auto unset = State::UNSET;
    if (!stream_id) {
        GXF_LOG_VERBOSE("Cuda stream_id not present for message eid:[E%05ld]", message->eid());
        GXF_ASSERT_TRUE(state_.compare_exchange_strong(unset, State::DATA_AVAILABLE));
        return GXF_SUCCESS;
    }

    // received an invalid stream
    if (stream_id->is_null() || stream_id.value()->stream_cid == kNullUid) {
        GXF_LOG_ERROR("Received empty cudastreamid for message eid:[E%05ld]: %s",
                      message->eid(), stream_id->name());
        return GXF_FAILURE;
    }

    // access the cuda handle and register the callback
    gxf::Handle<gxf::CudaStream> gxf_cuda_stream = UNWRAP_OR_RETURN(
      Handle<CudaStream>::Create(stream_id.value().context(), stream_id.value()->stream_cid),
      "Failed to get CudaStream");

    cudaStream_t cuda_stream = UNWRAP_OR_RETURN(gxf_cuda_stream.get()->stream(),
                                                "Failed to get cudaStream_t");
    GXF_LOG_VERBOSE("Registering callback for message eid:[E%05ld]", message->eid());
    GXF_ASSERT_TRUE(state_.compare_exchange_strong(unset, State::CALLBACK_REGISTERED));
    cudaError_t cuda_result = cudaLaunchHostFunc(cuda_stream, cudaHostCallback, this);
    CHECK_CUDA_ERROR_RESULT(cuda_result,
      "Unable to register host function using cudaLaunchHostFunc");
  }
  return GXF_SUCCESS;
}

gxf_result_t CudaStreamSchedulingTerm::check_abi(int64_t timestamp,
                                                 SchedulingConditionType* type,
                                                 int64_t* target_timestamp) const {
  auto current_state = state_.load();
  switch (current_state) {
    case State::DATA_AVAILABLE: {
      *type = SchedulingConditionType::READY;
      *target_timestamp = timestamp;
    } break;
    case State::CALLBACK_REGISTERED: {
      *type = SchedulingConditionType::WAIT_EVENT;
    } break;
    case State::UNSET: {
      GXF_LOG_VERBOSE("No messages to process for entity: E[%05ld]", eid());
      *type = SchedulingConditionType::WAIT;
    } break;
    default: {
      return GXF_FAILURE;
    }break;
  }

  return GXF_SUCCESS;
}

gxf_result_t CudaStreamSchedulingTerm::onExecute_abi(int64_t timestamp) {
  auto has_message = receiver_->size() > 0 ? true : false;
  auto current_state = state_.load();
  if (current_state == State::DATA_AVAILABLE) {
    // Check if message has been consumed by the entity
    if (!has_message || message_eid_ != receiver_->peek(0).value().eid()) {
      state_ = State::UNSET;
      message_eid_ = kNullUid;
    }
  }

  return GXF_SUCCESS;
}

gxf_result_t CudaEventSchedulingTerm::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      receiver_, "receiver", "Receiver queue",
      "The receiver queue on which the scheduling term checks for the data"
      "readiness on cuda stream based on the cuda event");
  result &= registrar->parameter(
      event_name_, "event_name", "Event name",
      "The event name on which the cudaEventQuery API is called to get the status",
      std::string(""));

  return ToResultCode(result);
}

gxf_result_t CudaEventSchedulingTerm::initialize() {
  current_state_ = SchedulingConditionType::WAIT;
  return GXF_SUCCESS;
}

gxf_result_t CudaEventSchedulingTerm::update_state_abi(int64_t timestamp) {
  Expected<Entity> message = Unexpected{GXF_FAILURE};

  // Sync any messages from the backstage before peeking into the messages
  auto code = receiver_->sync();
  if (!code) {
    return ToResultCode(code);
  }

  message = receiver_->peek(0);
  if (!message || message.value().is_null()) {
    current_state_ = SchedulingConditionType::WAIT;
    return GXF_SUCCESS;
  }

  auto maybe_event = message->get<CudaEvent>(event_name_.get().c_str());
  cudaEvent_t cuda_event{};
  if (maybe_event) {
    cuda_event = maybe_event.value().get()->event().value();
  } else {
    current_state_ = SchedulingConditionType::WAIT;
    return GXF_SUCCESS;
  }

  cudaError_t error{cudaSuccess};
  if ((error = cudaEventQuery(cuda_event)) == cudaErrorNotReady) {
    GXF_LOG_DEBUG("Data not yet ready cuda_error: %s, error_str: %s",
                  cudaGetErrorName(error), cudaGetErrorString(error));
    current_state_ = SchedulingConditionType::WAIT;
    return GXF_SUCCESS;
  }

  if (current_state_ != SchedulingConditionType::READY) {
    current_state_ =  SchedulingConditionType::READY;
    last_state_change_ = timestamp;
  }

  return GXF_SUCCESS;
}

gxf_result_t CudaEventSchedulingTerm::check_abi(int64_t timestamp,
                                                SchedulingConditionType* type,
                                                int64_t* target_timestamp) const {
  *type =  current_state_;
  *target_timestamp = last_state_change_;
  return GXF_SUCCESS;
}

gxf_result_t CudaEventSchedulingTerm::onExecute_abi(int64_t timestamp) {
  current_state_ = SchedulingConditionType::WAIT;
  return GXF_SUCCESS;
}


gxf_result_t CudaBufferAvailableSchedulingTerm::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      receiver_, "receiver", "Receiver queue",
      "The receiver queue on which the scheduling term checks for the data"
      "readiness on a cuda buffer");
  return ToResultCode(result);
}

void CUDART_CB CudaBufferAvailableSchedulingTerm::cudaHostCallback(void* term_ptr) {
  if (term_ptr == nullptr) {
    GXF_LOG_ERROR("Invalid data ptr provided for cudaHostCallback "
                  "in CudaBufferAvailableSchedulingTerm");
    return;
  }
  auto term = reinterpret_cast<CudaBufferAvailableSchedulingTerm*>(term_ptr);
  GXF_LOG_VERBOSE("Received host callback from cuda buffer for entity [E%05" PRId64 "]",
                  term->eid());
  auto cb_registered = State::CALLBACK_REGISTERED;
  GXF_ASSERT_TRUE(term->state_.compare_exchange_strong(cb_registered, State::DATA_AVAILABLE));
  GxfEntityEventNotify(term->receiver_->context(), term->receiver_->eid());
}

gxf_result_t CudaBufferAvailableSchedulingTerm::update_state_abi(int64_t timestamp) {
  Expected<Entity> message = Unexpected{GXF_FAILURE};
  auto current_state = state_.load();
  // cuda buffer wait is over, message can be dequeued
  if (current_state == State::DATA_AVAILABLE || current_state == State::CALLBACK_REGISTERED) {
    return GXF_SUCCESS;
  }

  // scheduling term is currently not monitoring any message, check for new incoming messages
  if (current_state == State::UNSET) {
    // Sync any messages from the backstage before peeking into the messages
    if (receiver_->size() == 0) {
      auto code = receiver_->sync();
      if (!code) { return ToResultCode(code); }
    }

    // no new messages, no change in state
    if (receiver_->size() == 0) { return GXF_SUCCESS; }

    auto message = receiver_->peek(0);
    if (!message || message.value().is_null()) {
      GXF_LOG_ERROR("Received invalid message at receiver [C%05ld] in entity [E%05ld]",
                    receiver_->cid(), eid());
      return GXF_FAILURE;
    }

    message_eid_ = message.value().eid();
    auto maybe_buffer = message->get<CudaBuffer>();
    // message without a cuda buffer component, no action taken
    // message must be consumed before we can process the next one in the queue
    auto unset = State::UNSET;
    if (!maybe_buffer) {
        GXF_LOG_ERROR("Cuda Buffer not present for message eid:[E%05ld]", message->eid());
        GXF_ASSERT_TRUE(state_.compare_exchange_strong(unset, State::DATA_AVAILABLE));
        return GXF_SUCCESS;
    }

    auto buffer = maybe_buffer.value();
    auto state = buffer->state();
    if (state == CudaBuffer::State::DATA_AVAILABLE) {
      GXF_LOG_VERBOSE("Skipping callback registration since data is already available");
      state_ = State::DATA_AVAILABLE;
      return GXF_SUCCESS;
    }

    GXF_ASSERT_TRUE(state_.compare_exchange_strong(unset, State::CALLBACK_REGISTERED));
    auto code = buffer->registerCallbackOnStream(cudaHostCallback, reinterpret_cast<void*>(this));
    if (!code) {
      GXF_LOG_ERROR("Unable to register host function using cuda buffer registerCallbackOnStream");
      return ToResultCode(code);
    }
  }
  return GXF_SUCCESS;
}

gxf_result_t CudaBufferAvailableSchedulingTerm::check_abi(int64_t timestamp,
                                                 SchedulingConditionType* type,
                                                 int64_t* target_timestamp) const {
  auto current_state = state_.load();
  switch (current_state) {
    case State::DATA_AVAILABLE: {
      *type = SchedulingConditionType::READY;
      *target_timestamp = timestamp;
    } break;
    case State::CALLBACK_REGISTERED: {
      *type = SchedulingConditionType::WAIT_EVENT;
    } break;
    case State::UNSET: {
      GXF_LOG_VERBOSE("No messages to process for entity: E[%05ld]", eid());
      *type = SchedulingConditionType::WAIT;
    } break;
    default: {
      return GXF_FAILURE;
    }break;
  }

  return GXF_SUCCESS;
}

gxf_result_t CudaBufferAvailableSchedulingTerm::onExecute_abi(int64_t timestamp) {
  auto has_message = receiver_->size() > 0 ? true : false;
  auto current_state = state_.load();
  if (current_state == State::DATA_AVAILABLE) {
    // Check if message has been consumed by the entity
    if (!has_message || message_eid_ != receiver_->peek(0).value().eid()) {
      state_ = State::UNSET;
      message_eid_ = kNullUid;
    }
  }

  return GXF_SUCCESS;
}

}  //  namespace gxf
}  //  namespace nvidia
