/*
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "ucx_receiver.hpp"

#include <arpa/inet.h> /* inet_addr */
#include <unistd.h>

#include <list>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "cuda_runtime.h"
#include "gxf/core/component.hpp"
#include "gxf/core/entity.hpp"
#include "gxf/std/timestamp.hpp"
#include "ucx_common.hpp"


namespace nvidia {

namespace gxf {


const char* UcxReceiver::get_addr() {
  return address_.get().c_str();
}

int UcxReceiver::get_port() {
  return port_;
}


/**
 * The callback on the receiving side, which is invoked upon receiving the
 * active message.
 */
static void am_recv_cb(void* request, ucs_status_t status, size_t length,
                       void* user_data) {
  test_req_t* ctx;
  if (user_data == NULL) {
      GXF_LOG_ERROR("user_data passed to am_recv_cb mustn't be NULL");
      return;
  }

  ctx           = static_cast<test_req_t*>(user_data);
  ctx->complete = 1;
}

gxf_result_t UcxReceiver::request_finalize_sync(ucp_worker_h ucp_worker, test_req_t* request,
                                                test_req_t* ctx) {
  ucs_status_t status;
  status = request_wait(ucp_worker, request, ctx);

  if (status != UCS_OK) {
      GXF_LOG_ERROR("Unable to receive UCX message (%s)", ucs_status_string(status));
      return GXF_FAILURE;
  }

  queue_->sync();

  am_data_desc_->receiving_message = false;
  return GXF_SUCCESS;
}


/**
 * Send and receive a message using Active Message API.
 * The client sends a message to the server and waits until the send is completed.
 * The server gets a message from the client and if it is rendezvous request,
 * initiates receive operation.
 */
gxf_result_t UcxReceiver::receive_message() {
  if (!cpu_data_only_) {
    cudaError_t error{cudaSuccess};
    error = cudaSetDevice(dev_id_);
    if (error != cudaSuccess) {
      GXF_LOG_ERROR("cudaSetDevice Failed - %d", error);
      return GXF_FAILURE;
    }
  }
  test_req_t *request;
  ucp_request_param_t params;
  test_req_t* ctx = new test_req_t();

  ctx->complete       = 0;
  ctx->header         = NULL;
  params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                        UCP_OP_ATTR_FIELD_DATATYPE |
                        UCP_OP_ATTR_FIELD_USER_DATA |
                        UCP_OP_ATTR_FLAG_NO_IMM_CMPL |
                        UCP_OP_ATTR_FIELD_MEMORY_TYPE;
  params.datatype = am_data_desc_->num_of_comps == 1 ? ucp_dt_make_contig(1) : UCP_DATATYPE_IOV;
  params.memory_type     = am_data_desc_->mem_type;
  params.user_data = ctx;
  params.cb.recv_am    = am_recv_cb;
  request              = static_cast<test_req_t*>(ucp_am_recv_data_nbx(ucp_worker_,
                                                            am_data_desc_->desc,
                                                            am_data_desc_->recv_buf,
                                                            am_data_desc_->msg_length,
                                                            &params));
  if (enable_async_) {
    requests.push_back({request, ctx});
    return GXF_SUCCESS;
  } else {
    return request_finalize_sync(ucp_worker_, request, ctx);
  }
}

gxf_result_t UcxReceiver::init_context(ucp_worker_h  ucp_worker,
                                       ucx_am_data_desc* am_data_desc,
                                       int fd,
                                       bool cpu_data_only,
                                       bool enable_async) {
  ucp_worker_ = ucp_worker;
  am_data_desc_ = am_data_desc;
  efd_signal_ = fd;
  cpu_data_only_ = cpu_data_only;
  enable_async_ = enable_async;
  return GXF_SUCCESS;
}

gxf_result_t UcxReceiver::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(capacity_, "capacity", "Capacity", "", 10UL);
  result &= registrar->parameter(policy_, "policy", "Policy", "0: pop, 1: reject, 2: fault", 2UL);
  result &= registrar->parameter(address_, "address", "Listener Address", "Address to listen on",
                                 std::string("0.0.0.0"));
  result &= registrar->parameter(port_, "port", "rx_port", "RX port", DEFAULT_UCX_PORT);
  result &= registrar->parameter(buffer_, "buffer", "Serialization Buffer", "");
  result &= registrar->resource(gpu_device_, "Optional GPU device resource");
  return ToResultCode(result);
}

Expected<void> UcxReceiver::set_port(int port) {
  return port_.set(port);
}

Expected<void> UcxReceiver::set_serialization_buffer(Handle<UcxSerializationBuffer> buffer) {
  if (buffer.is_null()) { return Unexpected{GXF_ARGUMENT_NULL}; }
  return buffer_.set(buffer);
}

gxf_result_t UcxReceiver::initialize() {
  if (cpu_data_only_) {
    GXF_LOG_INFO(
        "UcxReceiver [cid: %ld]: CPU-only mode selected. No GPUDevice resource will be used.",
        cid());
  } else if (gpu_device_.try_get()) {
    dev_id_ = gpu_device_.try_get().value()->device_id();
    GXF_LOG_INFO("Ucx Receiver [cid: %ld]: GPUDevice value found and cached. "
                 "dev_id: %d", cid(), dev_id_);
  }
  if (capacity_ == 0) {
    return GXF_ARGUMENT_OUT_OF_RANGE;
  }
  queue_ = std::make_unique<queue_t>(
      capacity_, (::gxf::staging_queue::OverflowBehavior)(policy_.get()), Entity{});
  return GXF_SUCCESS;
}

gxf_result_t UcxReceiver::deinitialize() {
  // Empty queue
  if (!queue_) {
    GXF_LOG_ERROR("Bad Queue in UcxReceiver with name '%s' and cid [C%05zu]",
                name(), cid());
    return GXF_CONTRACT_INVALID_SEQUENCE;
  }
  queue_->popAll();
  queue_->sync();
  queue_->popAll();
  return GXF_SUCCESS;
}

gxf_result_t UcxReceiver::peek_abi(gxf_uid_t* uid, int32_t index) {
  if (uid == nullptr) { return GXF_ARGUMENT_NULL; }
  if (!queue_) { return GXF_FAILURE; }

  *uid = queue_->peek(index).eid();
  if (*uid == kNullUid) { return GXF_FAILURE; }

  return GXF_SUCCESS;
}

gxf_result_t UcxReceiver::peek_back_abi(gxf_uid_t* uid, int32_t index) {
  if (uid == nullptr) { return GXF_ARGUMENT_NULL; }
  if (!queue_) { return GXF_FAILURE; }

  *uid = queue_->peek_backstage(index).eid();
  if (*uid == kNullUid) { return GXF_FAILURE; }

  return GXF_SUCCESS;
}

size_t UcxReceiver::capacity_abi() {
  return queue_ ? queue_->capacity() : 0;
}

size_t UcxReceiver::size_abi() {
  return queue_ ? queue_->size() : 0;
}

gxf_result_t UcxReceiver::receive_abi(gxf_uid_t* uid) {
  return pop_abi(uid);
}

gxf_result_t UcxReceiver::pop_abi(gxf_uid_t* uid) {
  if (uid == nullptr) { return GXF_ARGUMENT_NULL; }
  if (!queue_) {
    GXF_LOG_ERROR("Bad Queue in UcxReceiver with name '%s' and cid [C%05zu]",
                  name(), cid());
    return GXF_FAILURE;
  }

  Entity entity = queue_->pop();
  if (entity.is_null()) {
    GXF_LOG_VERBOSE("Received null entity in UcxReceiver with name '%s' cid [C%05zu]",
                    name(), cid());
    return GXF_FAILURE;
  }

  // We do not want to decrement the ref count (which will happen in the Entity destructor)
  // as we expect the caller to do that.
  const gxf_result_t code = GxfEntityRefCountInc(context(), entity.eid());
  if (code != GXF_SUCCESS) { return code; }

  *uid = entity.eid();
  return GXF_SUCCESS;
}

size_t UcxReceiver::back_size_abi() {
  return queue_ ? queue_->back_size() : 0;
}

gxf_result_t UcxReceiver::sync_abi() {
  return GXF_SUCCESS;
}

gxf_result_t UcxReceiver::sync_io_abi() {
  if (!queue_) { return GXF_FAILURE; }
  if (!ucp_worker_ || !am_data_desc_->receiving_message) return GXF_SUCCESS;

  return receive_message();
}

gxf_result_t UcxReceiver::wait_abi() {
    if (!enable_async_) {
      return GXF_SUCCESS;
    }
    gxf_result_t result;
    gxf_result_t ret = GXF_SUCCESS;
    for (auto it = requests.begin(); it != requests.end(); ) {
        // Dereference the iterator to get to the pair
        auto& pair = *it;
        result = request_finalize(ucp_worker_, pair.first, pair.second);
        if (result == GXF_SUCCESS) {
            it = requests.erase(it);
        } else if (result == GXF_NOT_FINISHED) {
            ++it;
            ret = GXF_NOT_FINISHED;
        } else {
            return GXF_FAILURE;
        }
    }
    if (requests.empty()) {
        queue_->sync();
        am_data_desc_->receiving_message = false;
        uint64_t val = 1;
        if (write(efd_signal_, &val, sizeof(val)) == -1) {
            GXF_LOG_ERROR("failed to signal UcxContext to exit wait");
        }
    }
    return ret;
}

gxf_result_t UcxReceiver::push_abi(gxf_uid_t other) {
  if (!queue_) { return GXF_FAILURE; }

  auto maybe = Entity::Shared(context(), other);
  if (!maybe) { return maybe.error(); }

  if (!queue_->push(std::move(maybe.value()))) {
    GXF_LOG_WARNING("Push failed on '%s'", name());
    return GXF_EXCEEDING_PREALLOCATED_SIZE;
  }

  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
