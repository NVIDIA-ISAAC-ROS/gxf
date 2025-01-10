/*
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "ucx_transmitter.hpp"

#include <arpa/inet.h> /* inet_addr */
#include <list>
#include <memory>
#include <queue>
#include <string>
#include <utility>

#include "gxf/multimedia/audio.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/eos.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"
#include "ucx_common.hpp"


#define IP_STRING_LEN          50
#define PORT_STRING_LEN        8

namespace nvidia {

namespace gxf {


/**
 * Error handling callback.
 */
static void ep_err_cb(void* arg, ucp_ep_h ep, ucs_status_t status) {
    if (status == UCS_ERR_CONNECTION_RESET) {
        GXF_LOG_DEBUG("client endpoint error handling callback was invoked with status %d (%s)",
                status, ucs_status_string(status));
    } else if (status != UCS_ERR_NOT_CONNECTED) {
        GXF_LOG_ERROR("client endpoint error handling callback was invoked with status %d (%s)",
                status, ucs_status_string(status));
    }
    bool* connection_closed_p = static_cast<bool*>(arg);
    *connection_closed_p = true;
}

/**
 * The callback on the sending side, which is invoked after finishing sending
 * the message.
 */
static void send_cb(void* request, ucs_status_t status, void* user_data) {
    test_req_t* ctx;
    if (user_data == NULL) {
        GXF_LOG_ERROR("user_data passed mustn't be NULL");
        return;
    }

    ctx = static_cast<test_req_t*>(user_data);
    ctx->complete = 1;
}

gxf_result_t UcxTransmitter::registerInterface(Registrar* registrar) {
    Expected<void> result;
    result &= registrar->parameter(capacity_, "capacity", "Capacity", "", 1UL);
    result &= registrar->parameter(policy_, "policy", "Policy", "0: pop, 1: reject, 2: fault", 2UL);
    result &= registrar->parameter(receiver_address_, "receiver_address", "Receiver address",
            "Address to connect to", std::string("0.0.0.0"));
    result &= registrar->parameter(port_, "port", "Receiver Port", "Receiver Port",
                                   DEFAULT_UCX_PORT);
    result &= registrar->parameter(local_address_, "local_address", "Local address",
            "Local Address to use for connection", std::string("0.0.0.0"));
    result &= registrar->parameter(local_port_, "local_port", "Local port",
            "Local Port to use for connection", 0U);
    result &= registrar->parameter(buffer_, "buffer", "Serialization Buffer", "");
    result &= registrar->parameter(maximum_connection_retries_, "maximum_connection_retries",
            "Maximum Connection Retries", "", 200U);
    result &= registrar->resource(gpu_device_, "Optional GPU device resource");
    return ToResultCode(result);
}

Expected<void> UcxTransmitter::set_port(int port) {
  return port_.set(port);
}

Expected<void> UcxTransmitter::set_serialization_buffer(Handle<UcxSerializationBuffer> buffer) {
  if (buffer.is_null()) { return Unexpected{GXF_ARGUMENT_NULL}; }
  return buffer_.set(buffer);
}

gxf_result_t UcxTransmitter::initialize() {
    if (cpu_data_only_) {
      GXF_LOG_INFO(
          "UcxTransmitter [cid: %ld]: CPU-only mode selected. No GPUDevice resource will be used.",
          cid());
    } else if (gpu_device_.try_get()) {
        dev_id_ = gpu_device_.try_get().value()->device_id();
        GXF_LOG_INFO("Ucx Transmitter [cid: %ld]: GPUDevice value found and cached. "
                "dev_id: %d", cid(), dev_id_);
    }
    if (capacity_ == 0) {
        return GXF_ARGUMENT_OUT_OF_RANGE;
    }
    queue_ = std::make_unique<queue_t>(
            capacity_, (::gxf::staging_queue::OverflowBehavior)(policy_.get()), Entity{});
    index = 0;
    return GXF_SUCCESS;
}

gxf_result_t UcxTransmitter::deinitialize() {
    if (!queue_) {
        GXF_LOG_ERROR("Bad Queue in UcxTransmitter");
        return GXF_CONTRACT_INVALID_SEQUENCE;
    }
    queue_->popAll();
    queue_->sync();
    queue_->popAll();
    return GXF_SUCCESS;
}

void blocking_ep_flush(ucp_worker_h worker, ucp_ep_h ep) {
    ucp_request_param_t param;
    void *request;

    param.op_attr_mask = 0;
    request = ucp_ep_flush_nbx(ep, &param);
    process_request(worker, request);
    return;
}

/**
 * Initialize the client side. Create an endpoint from the client side to be
 * connected to the remote server (to the given IP).
 */
gxf_result_t UcxTransmitter::create_client_connection() {
    ucp_ep_params_t ep_params;
    struct sockaddr_storage connect_addr;
    struct sockaddr_storage local_connect_addr;
    ucs_status_t status;

    set_sock_addr(receiver_address_.get().c_str(), port_, &connect_addr);
    set_sock_addr(local_address_.get().c_str(), local_port_, &local_connect_addr);
    ep_params.field_mask       = UCP_EP_PARAM_FIELD_FLAGS       |
                                 UCP_EP_PARAM_FIELD_SOCK_ADDR   |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLER |
                                 UCP_EP_PARAM_FIELD_LOCAL_SOCK_ADDR |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.err_handler.cb   = ep_err_cb;
    ep_params.err_handler.arg  = static_cast<void*>(connection_closed_p_);
    ep_params.flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr    = (struct sockaddr*)&connect_addr;
    ep_params.sockaddr.addrlen = sizeof(connect_addr);
    ep_params.local_sockaddr.addr = (struct sockaddr*)&local_connect_addr;
    ep_params.local_sockaddr.addrlen = sizeof(local_connect_addr);
    *connection_closed_p_ = false;
    status = ucp_ep_create(ucp_worker_, &ep_params, ep_);
    if (status != UCS_OK) {
        GXF_LOG_ERROR("Failed to connect to %s (%s)", receiver_address_.get().c_str(),
                      ucs_status_string(status));
        return GXF_FAILURE;
    }
    blocking_ep_flush(ucp_worker_, *ep_);
    if (*connection_closed_p_) {
        return GXF_SUCCESS;
    }
    GXF_LOG_INFO("Connection established");
    return GXF_SUCCESS;
}


gxf_result_t UcxTransmitter::check_connection_and_connect() {
    gxf_result_t result;
    if (!ucp_worker_) {
        GXF_LOG_ERROR("UCP worker has not been initialized.");
        return GXF_FAILURE;
    }
    ucp_worker_progress(ucp_worker_);
    if (!*connection_closed_p_) {
        return GXF_SUCCESS;
    }
    if (reconnect_) {
        GXF_LOG_WARNING("Connection closed on send. Trying to reconnect...");
        result = create_client_connection_with_retries();
        if (result != GXF_SUCCESS) {
            return result;
        }
    } else {
        GXF_LOG_ERROR("Connection is found closed during send attempt.");
        return GXF_FAILURE;
    }
    return GXF_SUCCESS;
}

/**
 * Send using Active Message API.
 * The client sends a message to the server and waits until the send is completed.
 */
gxf_result_t UcxTransmitter::send_am(Entity& entity) {
    gxf_result_t result;
    if (!cpu_data_only_) {
        cudaError_t error{cudaSuccess};
        error = cudaSetDevice(dev_id_);
        if (error != cudaSuccess) {
            GXF_LOG_ERROR("cudaSetDevice Failed - %d, device id %d", error, dev_id_);
            return GXF_FAILURE;
        }
    }
    result = check_connection_and_connect();
    if (result != GXF_SUCCESS) {
        return result;
    }

    ucp_request_param_t params;
    size_t msg_length;
    void* msg;
    test_req_t* ctx = new test_req_t();

    buffer_.get()->reset();
    // header
    auto header_size = entity_serializer_->serializeEntity(entity, buffer_.get());
    if (!header_size) {
        GXF_LOG_ERROR("Serialization failed");
        return GXF_FAILURE;
    }
    size_t num_of_comps = buffer_->iov_buffer_size();

    ucp_dt_iov_t* comp_data = const_cast<ucp_dt_iov_t*>(buffer_->iov_buffer());
    msg = (num_of_comps == 1) ? comp_data[0].buffer : comp_data;
    msg_length = (num_of_comps == 1) ? comp_data[0].length : num_of_comps;
    ctx->complete = 0;

    /* Client sends a message to the server using the AM API */
    params.op_attr_mask =  UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA |
                           UCP_OP_ATTR_FIELD_FLAGS | UCP_OP_ATTR_FIELD_DATATYPE |
                           UCP_OP_ATTR_FIELD_MEMORY_TYPE;
    params.user_data    =  ctx;
    params.datatype     =  (num_of_comps == 1) ? ucp_dt_make_contig(1) : UCP_DATATYPE_IOV;
    params.memory_type  =  buffer_->mem_type();
    params.flags        =  UCP_AM_SEND_FLAG_RNDV;
    params.cb.send      =  (ucp_send_nbx_callback_t)send_cb;
    if (enable_async_) {
        void* request;
        char* header_copy = new char[header_size.value()];
        std::memcpy(header_copy, buffer_.get()->data(), header_size.value());
        ctx->header = header_copy;
        request             =  static_cast<void*>(ucp_am_send_nbx(*ep_, TEST_AM_ID,
                                                                        header_copy,
                                                                        header_size.value(), msg,
                                                                        msg_length, &params));
        {
            std::lock_guard<std::mutex> lock(*mtx_);
            send_queue_->push_back({entity, ucp_worker_, request, ctx, ++index});
        }
        cv_->notify_one();
    } else {
        test_req_t* request;
        ucs_status_t status;
        request             =  static_cast<test_req_t*>(ucp_am_send_nbx(*ep_, TEST_AM_ID,
                                                                        buffer_.get()->data(),
                                                                        header_size.value(), msg,
                                                                        msg_length, &params));


        status = request_wait(ucp_worker_, request, ctx);
        if (status != UCS_OK) {
            GXF_LOG_ERROR("unable to send UCX message (%s)",
                    ucs_status_string(status));
            return GXF_FAILURE;
        }
    }
    return GXF_SUCCESS;
}

gxf_result_t UcxTransmitter::sync_abi() {
    return GXF_SUCCESS;
}

gxf_result_t UcxTransmitter::sync_io_abi() {
    if (!queue_) {
        GXF_LOG_ERROR("No QUEUE");
        return GXF_FAILURE;
    }

    // Move the message to mainstage from backstage
    if (!queue_->sync()) {
        GXF_LOG_WARNING("Sync failed on '%s'", name());
        return GXF_EXCEEDING_PREALLOCATED_SIZE;
    }
    // Pop the message from the mainstage
    Entity entity = queue_->pop();
    if (entity.is_null()) {
        GXF_LOG_WARNING("Received null entity in UcxTransmitter with name '%s' cid [C%05zu]",
                        name(), cid());
        return GXF_SUCCESS;
    }
    if (send_am(entity) != GXF_SUCCESS) {
        GXF_LOG_ERROR("Failed to send entity");
        return GXF_FAILURE;
    }

    return GXF_SUCCESS;
}


gxf_result_t UcxTransmitter::create_client_connection_with_retries() {
    gxf_result_t result;
    uint32_t connection_retries = 0;
    // set up timer interval
    int interval_sec = 1;
    auto start_time = std::chrono::steady_clock::now();
    while ((*connection_closed_p_) && (connection_retries < maximum_connection_retries_)) {
        // check if timer interval has elapsed
        auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time).count();
        if (elapsed_time >= interval_sec) {
                // attempt connection
                result = create_client_connection();
                if (result != GXF_SUCCESS) {
                    return GXF_FAILURE;
                }
                // check if connection is successful
                if (!*connection_closed_p_) {
                    break;
                }
                GXF_LOG_DEBUG("Failed to connect to IP '%s' retry #%d",
                        receiver_address_.get().c_str(), connection_retries);
                // reset timer
                start_time = std::chrono::steady_clock::now();
                connection_retries++;
            }
    }
    if (*connection_closed_p_) {
        GXF_LOG_ERROR("Failed to establish connection");
        return GXF_FAILURE;
    }
    return GXF_SUCCESS;
}

gxf_result_t UcxTransmitter::init_context(ucp_context_h ucp_context,
                                          Handle<EntitySerializer> serializer,
                                          ucp_worker_h ucp_worker,
                                          ucp_ep_h* ep,
                                          bool* connection_closed_p,
                                          bool reconnect,
                                          bool cpu_data_only,
                                          bool enable_async,
                                          std::list<UcxTransmitterSendContext_>* send_queue,
                                          std::condition_variable* cv,
                                          std::mutex* mtx) {
    if (ucp_context == NULL) {
        GXF_LOG_ERROR("ucp context is NULL");
        return GXF_FAILURE;
    }
    if (!serializer.try_get()) {
        GXF_LOG_ERROR("EntitySerializer is NULL");
        return GXF_FAILURE;
    } else {
        entity_serializer_ = serializer;
    }
    if (enable_async_ && !send_queue) {
        GXF_LOG_ERROR("send queue is NULL");
        return GXF_FAILURE;
    }
    cv_ = cv;
    mtx_ = mtx;
    send_queue_ = send_queue;
    ucp_worker_ = ucp_worker;
    ep_ = ep;
    connection_closed_p_ = connection_closed_p;
    reconnect_ = reconnect;
    enable_async_ = enable_async;
    cpu_data_only_ = cpu_data_only;
    return create_client_connection_with_retries();
}

gxf_result_t UcxTransmitter::peek_abi(gxf_uid_t* uid, int32_t index) {
  if (uid == nullptr) { return GXF_ARGUMENT_NULL; }
  if (!queue_) { return GXF_FAILURE; }

  *uid = queue_->peek(index).eid();
  if (*uid == kNullUid) { return GXF_FAILURE; }

  return GXF_SUCCESS;
}

size_t UcxTransmitter::capacity_abi() {
  return queue_ ? queue_->capacity() : 0;
}

size_t UcxTransmitter::size_abi() {
  return queue_ ? queue_->size() : 0;
}

gxf_result_t UcxTransmitter::publish_abi(gxf_uid_t uid) {
  return push_abi(uid);
}

size_t UcxTransmitter::back_size_abi() {
  return queue_ ? queue_->back_size() : 0;
}

gxf_result_t UcxTransmitter::pop_abi(gxf_uid_t* uid) {
  return GXF_SUCCESS;
}

gxf_result_t UcxTransmitter::pop_io_abi(gxf_uid_t* uid) {
  if (uid == nullptr) { return GXF_ARGUMENT_NULL; }
  if (!queue_) {
    GXF_LOG_ERROR("Bad Queue in UcxTransmitter");
    return GXF_FAILURE;
  }

  Entity entity = queue_->pop();
  if (entity.is_null()) {
      GXF_LOG_ERROR("Received null entity in ucx transmitter");
    return GXF_FAILURE;
  }

  // We do not want to decrement the ref count (which will happen in the Entity destructor)
  // as we expect the caller to do that.
  const gxf_result_t code = GxfEntityRefCountInc(context(), entity.eid());
  if (code != GXF_SUCCESS) { return code; }

  *uid = entity.eid();
  return GXF_SUCCESS;
}

gxf_result_t UcxTransmitter::push_abi(gxf_uid_t other) {
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
