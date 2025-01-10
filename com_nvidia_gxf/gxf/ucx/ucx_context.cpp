/*
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <list>
#include <memory>
#include <utility>
#include <vector>

#include "ucx_common.hpp"
#include "ucx_context.hpp"

#include "gxf/core/component.hpp"
#include "gxf/multimedia/audio.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/eos.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"

#define IP_STRING_LEN          50
#define PORT_STRING_LEN        8
#define WORKER_PROGRESS_RETRIES        5
namespace nvidia {
namespace gxf {

static gxf_result_t add_epoll_fd(int epoll_fd, int fd);
static gxf_result_t del_epoll_fd(int epoll_fd, int fd);

gxf_result_t UcxContext::registerInterface(Registrar* registrar) {
    Expected<void> result;
    result &= registrar->parameter(entity_serializer_, "serializer", "Entity Serializer", "");
    result &= registrar->resource(gpu_device_, "Optional GPU device resource");
    result &= registrar->parameter(reconnect_, "reconnect", "Reconnect",
            "Try to reconnect if a connection is closed during run", true);
    result &= registrar->parameter(cpu_data_only_, "cpu_data_only", "CPU data only",
            "If true, the UCX context will only support transmission of CPU (host) data.", false);
    result &= registrar->parameter(enable_async_, "enable_async",
            "enable asynchronous transmit/receive",
            "If true, UCX transmit and receive will queue messages to be sent asynchronously.",
            true);
  return gxf::ToResultCode(result);
}

gxf_result_t UcxContext::initialize() {
  if (cpu_data_only_.get()) {
    GXF_LOG_INFO(
        "Ucx Context [cid: %ld]: CPU-only mode selected. No GPUDevice resource will be used.",
        cid());
  } else if (gpu_device_.try_get()) {
    dev_id_ = gpu_device_.try_get().value()->device_id();
    GXF_LOG_INFO(
        "Ucx Context [cid: %ld]: GPUDevice value found and cached. dev_id: %d.", cid(), dev_id_);
  }
  if (enable_async_.get()) {
    return GXF_SUCCESS;
  } else {
    close_server_loop_ = false;
    return init_context();
  }
}

gxf_result_t UcxContext::deinitialize() {
  if (enable_async_.get()) {
    if (rx_thread_.joinable()) {
        rx_thread_.join();
    }
    if (tx_thread_.joinable()) {
        tx_thread_.join();
    }
  } else {
    if (t_.joinable()) {
        t_.join();
    }
  }
    ucp_cleanup(ucp_context_);
    return GXF_SUCCESS;
}

void UcxContext::destroy_rx_contexts() {
    for (auto rx_context_iter : rx_contexts_) {
        auto rx_context = rx_context_iter.value();
        if (rx_context->conn_state == CONNECTED) {
            ep_close(rx_context->ucp_worker, rx_context->ep, 0);
        }
        rx_context->conn_state = CLOSED;
        if (rx_context->server_context.listener)
            ucp_listener_destroy(rx_context->server_context.listener);
        if (rx_context->server_context.listener_worker)
            ucp_worker_destroy(rx_context->server_context.listener_worker);
        if (rx_context->ucp_worker)
            ucp_worker_destroy(rx_context->ucp_worker);
    }
    rx_contexts_.clear();
}

void UcxContext::destroy_tx_contexts() {
    for (auto tx_context_iter : tx_contexts_) {
        auto tx_context = tx_context_iter.value();
        if (!tx_context->connection_closed) {
            ep_close(tx_context->ucp_worker, tx_context->ep, 0);
            tx_context->connection_closed = true;
        }
        ucp_worker_destroy(tx_context->ucp_worker);
    }
    tx_contexts_.clear();
}

Expected<void> UcxContext::removeRoutes(const Entity& entity) {
  if (tx_contexts_.size()) {
    if (enable_async_.get()) {
      {
          std::lock_guard<std::mutex> lock(mtx_);
          areTransmittersDone = true;
      }
      cv_.notify_one();
      tx_thread_.join();
    }
    destroy_tx_contexts();
  }
  if (rx_contexts_.size()) {
      close_server_loop_ = true;
      if (enable_async_.get()) {
        // Signal to wait_epoll
        uint64_t val = 1;
        if (write(efd_signal_, &val, sizeof(val)) == -1) {
            GXF_LOG_ERROR("Failed to signal thread to close");
        }
        rx_thread_.join();
      } else {
        t_.join();
      }
      close_server_loop_ = false;
      destroy_rx_contexts();
  }
  return Success;
}

Expected<void> UcxContext::addRoutes(const Entity& entity) {
  gxf_result_t result;
  auto transmitters = entity.findAllHeap<UcxTransmitter>();
  auto receivers = entity.findAllHeap<UcxReceiver>();
  for (auto rx : receivers.value()) {
      if (rx) {
          result = init_rx(rx.value());
          if (result != GXF_SUCCESS) {
              goto error_out;
          }
      }
  }
  if (enable_async_.get()) {
    if (rx_contexts_.size() && !rx_thread_.joinable()) {
        std::thread t([this] {start_server_async_queue();});
        rx_thread_ = std::move(t);
    }
  } else {
    if (rx_contexts_.size() && !t_.joinable()) {
        std::thread t([this] {start_server();});
        t_ = std::move(t);
    }
  }
  for (auto tx : transmitters.value()) {
      if (tx) {
          result = init_tx(tx.value());
          if (result != GXF_SUCCESS) {
              goto error_out;
          }
      }
  }
  if (enable_async_.get()) {
    if (tx_contexts_.size() && !tx_thread_.joinable()) {
        std::thread t([this] {poll_queue();});
        tx_thread_ = std::move(t);
    }
  }
  return Success;

error_out:
  removeRoutes(entity);
  return ExpectedOrCode(result);
}

void UcxContext::poll_queue() {
  if (!cpu_data_only_.get()) {
    cudaError_t error{cudaSuccess};
    error = cudaSetDevice(dev_id_);
    if (error != cudaSuccess) {
        GXF_LOG_ERROR("cudaSetDevice Failed - %d", error);
        return;
    }
  }

  gxf_result_t result;
  while (true) {
      std::unique_lock<std::mutex> lock(mtx_);
      if (areTransmittersDone && (pending_send_requests_.empty())) {
          lock.unlock();
          break;
      }
      cv_.wait(lock, [this]{ return (areTransmittersDone || !pending_send_requests_.empty()); });
      while (!pending_send_requests_.empty()) {
          // Process the queue entry
          for (auto it = pending_send_requests_.begin(); it != pending_send_requests_.end(); ) {
              // Dereference the iterator to get to the pair
              auto& data = *it;
              result = request_finalize(data.ucp_worker, data.request, data.ctx);
              if (result == GXF_SUCCESS) {
                  it = pending_send_requests_.erase(it);
              } else if (result == GXF_NOT_FINISHED) {
                  ++it;
              } else {
                  GXF_LOG_ERROR("exit tx thread");
                  return;
              }
              lock.unlock();
              lock.lock();
          }

          lock.unlock();
          lock.lock();
      }
      lock.unlock();
  }
}

static char* sockaddr_get_ip_str(const struct sockaddr_storage* sock_addr,
                                 char* ip_str, size_t max_size) {
  struct sockaddr_in  addr_in;
  struct sockaddr_in6 addr_in6;

  switch (sock_addr->ss_family) {
  case AF_INET:
      memcpy(&addr_in, sock_addr, sizeof(struct sockaddr_in));
      inet_ntop(AF_INET, &addr_in.sin_addr, ip_str, max_size);
      return ip_str;
  case AF_INET6:
      memcpy(&addr_in6, sock_addr, sizeof(struct sockaddr_in6));
      inet_ntop(AF_INET6, &addr_in6.sin6_addr, ip_str, max_size);
      return ip_str;
  default:
      return const_cast<char*>("Invalid address family");
  }
}

static char* sockaddr_get_port_str(const struct sockaddr_storage* sock_addr,
                                   char* port_str, size_t max_size) {
  struct sockaddr_in  addr_in;
  struct sockaddr_in6 addr_in6;

  switch (sock_addr->ss_family) {
  case AF_INET:
      memcpy(&addr_in, sock_addr, sizeof(struct sockaddr_in));
      snprintf(port_str, max_size, "%d", ntohs(addr_in.sin_port));
      return port_str;
  case AF_INET6:
      memcpy(&addr_in6, sock_addr, sizeof(struct sockaddr_in6));
      snprintf(port_str, max_size, "%d", ntohs(addr_in6.sin6_port));
      return port_str;
  default:
      return const_cast<char*>("Invalid address family");
  }
}

/**
 * The callback on the server side which is invoked upon receiving a connection
 * request from the client.
 */
static void server_conn_handle_cb(ucp_conn_request_h conn_request, void* arg) {
  UcxReceiverContext* rx_context = static_cast<UcxReceiverContext*>(arg);
  ucx_server_ctx_t* context = &rx_context->server_context;
  ucp_conn_request_attr_t attr;
  char ip_str[IP_STRING_LEN];
  char port_str[PORT_STRING_LEN];
  (void)ip_str;  // avoid unused variable warning
  (void)port_str;  // avoid unused variable warning
  ucs_status_t status;

  attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
  status = ucp_conn_request_query(conn_request, &attr);
  if (status != UCS_OK) {
      GXF_LOG_ERROR("Failed to query the connection request (%s)",
              ucs_status_string(status));
  } else {
      (void)sockaddr_get_ip_str;  // avoid unused variable warning
      (void)sockaddr_get_port_str;  // avoid unused variable warning
      GXF_LOG_INFO("Connection request received to %s:%d from client at address %s:%s",
              rx_context->rx->get_addr(), rx_context->rx->get_port(),
              sockaddr_get_ip_str(&attr.client_address, ip_str, sizeof(ip_str)),
              sockaddr_get_port_str(&attr.client_address, port_str, sizeof(port_str)));
  }
  if (context->conn_request == NULL) {
      context->conn_request = conn_request;
  } else {
      /* The server is already handling a connection request from a client,
        * reject this new one */
      GXF_LOG_ERROR("Rejecting a connection request. "
              "Only one client at a time is supported.");
      status = ucp_listener_reject(context->listener, conn_request);
      if (status != UCS_OK) {
          GXF_LOG_ERROR("Server failed to reject a connection request: (%s)",
                  ucs_status_string(status));
      }
  }
}

gxf_result_t UcxContext::init_tx(Handle<UcxTransmitter> tx) {
  gxf_result_t result;
  auto tx_context = std::make_shared<UcxTransmitterContext>();
  tx_context->tx = tx.get();
  tx_context->connection_closed = true;

  result = init_worker(ucp_context_, &tx_context->ucp_worker);
  if (result != GXF_SUCCESS) {
      return result;
  }
  tx_context->index = static_cast<int>(tx_contexts_.size());
  result = tx->init_context(ucp_context_, entity_serializer_.get(),
          tx_context->ucp_worker, &tx_context->ep,
          &tx_context->connection_closed, reconnect_.get(),
          cpu_data_only_.get(), enable_async_.get(), &pending_send_requests_,
          &cv_, &mtx_);
  if (result != GXF_SUCCESS) {
      goto destroy_worker;
  }
  tx_contexts_.push_back(tx_context);
  return GXF_SUCCESS;

destroy_worker:
  ucp_worker_destroy(tx_context->ucp_worker);
  return result;
}

void init_ucx_am_data_desc(ucx_am_data_desc* am_data_desc) {
  am_data_desc->complete = 0;
  am_data_desc->desc = 0;
  am_data_desc->msg_length = 0;
  am_data_desc->header_length = 0;
  am_data_desc->recv_buf = NULL;
  am_data_desc->header = NULL;
  am_data_desc->num_of_comps = 0;
  am_data_desc->receiving_message = false;
  am_data_desc->mem_type = UCS_MEMORY_TYPE_HOST;
}

/**
 * Initialize the server side. The server starts listening on the set address.
 */
gxf_result_t create_listener(std::shared_ptr<UcxReceiverContext> rx_context) {
  struct sockaddr_storage listen_addr;
  ucp_listener_params_t params;
  ucp_listener_attr_t attr;
  ucs_status_t status;
  char ip_str[IP_STRING_LEN];
  char port_str[PORT_STRING_LEN];
  (void)ip_str;  // avoid unused variable warning
  (void)port_str;  // avoid unused variable warning

  set_sock_addr(rx_context->rx->get_addr(), rx_context->rx->get_port(), &listen_addr);

  params.field_mask         = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                              UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
  params.sockaddr.addr      = (const struct sockaddr*)&listen_addr;
  params.sockaddr.addrlen   = sizeof(listen_addr);
  params.conn_handler.cb    = server_conn_handle_cb;
  params.conn_handler.arg   = rx_context.get();

  /* Create a listener on the server side to listen on the given address.*/
  status = ucp_listener_create(rx_context->server_context.listener_worker,
          &params, &rx_context->server_context.listener);
  if (status != UCS_OK) {
      GXF_LOG_ERROR("Failed to listen (%s)", ucs_status_string(status));
      return GXF_FAILURE;
  }

  /* Query the created listener to get the port it is listening on. */
  attr.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR;
  status = ucp_listener_query(rx_context->server_context.listener, &attr);
  if (status != UCS_OK) {
      GXF_LOG_ERROR("Failed to query the listener (%s)",
              ucs_status_string(status));
      ucp_listener_destroy(rx_context->server_context.listener);
      return GXF_FAILURE;
  }

  GXF_LOG_INFO("Server is listening on IP %s port %s. Waiting for connection...",
          sockaddr_get_ip_str(&attr.sockaddr, ip_str, IP_STRING_LEN),
          sockaddr_get_port_str(&attr.sockaddr, port_str, PORT_STRING_LEN));
  return GXF_SUCCESS;
}

ucs_status_t ucp_am_data_legacy_cb(void* arg, const void* header, size_t header_length,
                                   void* data, size_t length,
                                   const ucp_am_recv_param_t* param) {
  ucx_am_data_desc* am_desc;
  UcxReceiverContext* rx_context = static_cast<UcxReceiverContext*>(arg);
  am_desc = static_cast<ucx_am_data_desc*>(&rx_context->am_data_desc);
  am_desc->header = malloc(header_length);
  memcpy(am_desc->header, header, header_length);
  am_desc->header_length = header_length;
  am_desc->msg_length = length;
  am_desc->desc    = data;
  am_desc->complete = 1;
  return UCS_INPROGRESS;
}

ucs_status_t ucp_am_data_cb(void* arg, const void* header, size_t header_length,
                            void* data, size_t length,
                            const ucp_am_recv_param_t* param) {
  ucx_am_data_desc* am_desc;
  UcxReceiverContext* rx_context = static_cast<UcxReceiverContext*>(arg);
  am_desc = static_cast<ucx_am_data_desc*>(&rx_context->am_data_desc);
  if ((!am_desc->complete) && (rx_context->headers.size() == 0)) {
      am_desc->header = malloc(header_length);
      memcpy(am_desc->header, header, header_length);
      am_desc->header_length = header_length;
      am_desc->msg_length = length;
      am_desc->desc    = data;
      am_desc->complete = 1;
  } else {
      auto q_am_desc = std::make_shared<ucx_am_data_desc>();
      q_am_desc->header = malloc(header_length);
      memcpy(q_am_desc->header, header, header_length);
      q_am_desc->header_length = header_length;
      q_am_desc->msg_length = length;
      q_am_desc->desc    = data;
      q_am_desc->complete = 1;
      rx_context->headers.push_back(q_am_desc);
  }
  return UCS_INPROGRESS;
}

gxf_result_t
register_am_recv_callback(ucp_worker_h worker, std::shared_ptr<UcxReceiverContext> rx_context) {
  ucp_am_handler_param_t param;
  ucs_status_t status;
  param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                      UCP_AM_HANDLER_PARAM_FIELD_CB |
                      UCP_AM_HANDLER_PARAM_FIELD_ARG;
  param.id         = TEST_AM_ID;
  param.cb         = ucp_am_data_cb;
  param.arg        = rx_context.get();

  status = ucp_worker_set_am_recv_handler(worker, &param);
  if (status != UCS_OK) {
      GXF_LOG_ERROR("Failed to register server callback");
      return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t
register_am_recv_legacy_callback(ucp_worker_h worker,
                                 std::shared_ptr<UcxReceiverContext> rx_context) {
  GXF_LOG_INFO("UcxContext::register_am_recv_legacy_callback");
  ucp_am_handler_param_t param;
  ucs_status_t status;
  param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                      UCP_AM_HANDLER_PARAM_FIELD_CB |
                      UCP_AM_HANDLER_PARAM_FIELD_ARG;
  param.id         = TEST_AM_ID;
  param.cb         = ucp_am_data_legacy_cb;
  param.arg        = rx_context.get();

  status = ucp_worker_set_am_recv_handler(worker, &param);
  if (status != UCS_OK) {
      GXF_LOG_ERROR("Failed to register server callback");
      return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t UcxContext::epoll_add_worker(
  std::shared_ptr<UcxReceiverContext> rx_context, bool is_listener) {
  ucs_status_t status;
  ucp_worker_h worker = is_listener ?
          rx_context->server_context.listener_worker : rx_context->ucp_worker;
  int* worker_fd = is_listener ? &rx_context->server_context.listener_fd :
          &rx_context->worker_fd;


  if (epoll_fd_ == -1) {
      GXF_LOG_ERROR("failed to add worker to epoll: epoll_fd_ == -1");
      return GXF_SUCCESS;
  }

  status = ucp_worker_get_efd(worker, worker_fd);
  if (status != UCS_OK) {
      GXF_LOG_ERROR("failed to get ucp_worker fd to be epoll monitored");
      return GXF_FAILURE;
  }

  if (gxf_arm_worker(rx_context, is_listener) !=  GXF_SUCCESS) {
      GXF_LOG_ERROR("failed to arm fd %d to epoll", *worker_fd);
      return GXF_FAILURE;
  }

  if (add_epoll_fd(epoll_fd_, *worker_fd) != GXF_SUCCESS) {
      GXF_LOG_ERROR("failed to add fd %d to epoll", *worker_fd);
      return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}


gxf_result_t UcxContext::init_rx(Handle<UcxReceiver> rx) {
  gxf_result_t result;
  auto rx_context = std::make_shared<UcxReceiverContext>();
  rx_context->rx = rx.get();
  init_ucx_am_data_desc(&rx_context->am_data_desc);
  rx_context->conn_state = INIT;
  rx_conns_.total++;

  CHECK_RETURN_FAILURE(init_worker(ucp_context_,
          &rx_context->server_context.listener_worker));

  if (enable_async_.get()) {
    rx_conns_.total++;

    result = epoll_add_worker(rx_context, true);
    if (result != GXF_SUCCESS) {
        goto destroy_listener_worker;
    }
    rx_context->index = static_cast<int>(rx_contexts_.size());
    rx_context->server_context.conn_request = NULL;
    result = create_listener(rx_context);
    if (result != GXF_SUCCESS) {
        goto destroy_listener_worker;
    }
    rx_contexts_.push_back(rx_context);
    return GXF_SUCCESS;
  } else {
    result = init_worker(ucp_context_, &rx_context->ucp_worker);
    if (result != GXF_SUCCESS) {
        goto destroy_listener_worker;
    }
    result = rx_context->rx->init_context(rx_context->ucp_worker,
            &rx_context->am_data_desc, 0, cpu_data_only_.get(),
            enable_async_.get());
    if (result != GXF_SUCCESS) {
        goto destroy_data_worker;
    }
    rx_context->index = static_cast<int>(rx_contexts_.size());
    result = register_am_recv_legacy_callback(rx_context->ucp_worker, rx_context);
    if (result != GXF_SUCCESS) {
        goto deinit_context;
    }
    rx_context->server_context.conn_request = NULL;
    result = create_listener(rx_context);
    if (result != GXF_SUCCESS) {
        goto deinit_context;
    }
    rx_contexts_.push_back(rx_context);
    return GXF_SUCCESS;
  }

deinit_context:
    rx_context->rx->init_context(NULL, NULL, 0, false, false);
destroy_data_worker:
    ucp_worker_destroy(rx_context->ucp_worker);
destroy_listener_worker:
  ucp_worker_destroy(rx_context->server_context.listener_worker);
  return result;
}

static void err_cb(void* arg, ucp_ep_h ep, ucs_status_t status) {
  if (status == UCS_ERR_CONNECTION_RESET) {
    GXF_LOG_DEBUG("Server endpoint connection dropped with status %d (%s)",
            status, ucs_status_string(status));
  } else {
    GXF_LOG_ERROR("Server endpoint connection dropped with status %d (%s)",
            status, ucs_status_string(status));
  }
  ConnState* conn_state_p = static_cast<ConnState*>(arg);
  if (*conn_state_p == CONNECTED) {
      *conn_state_p = RESET;
  } else {
      *conn_state_p = CLOSED;
  }
}

gxf_result_t UcxContext::server_create_ep(std::shared_ptr<UcxReceiverContext> rx_context) {
  ucp_ep_params_t ep_params;
  ucs_status_t    status;
  ep_params.field_mask      = UCP_EP_PARAM_FIELD_ERR_HANDLER |
                              UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                              UCP_EP_PARAM_FIELD_CONN_REQUEST;
  ep_params.conn_request    = rx_context->server_context.conn_request;
  ep_params.err_mode        = UCP_ERR_HANDLING_MODE_PEER;
  ep_params.err_handler.cb  = err_cb;
  ep_params.err_handler.arg = static_cast<void*>(&rx_context->conn_state);
  if (rx_context->ucp_worker == NULL) {
      GXF_LOG_ERROR("data_worker is NULL");
      return GXF_FAILURE;
  }
  status = ucp_ep_create(rx_context->ucp_worker, &ep_params, &rx_context->ep);
  if (status != UCS_OK) {
      GXF_LOG_ERROR("Failed to create an endpoint on the server: (%s)",
              ucs_status_string(status));
      return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t UcxContext::init_connection(std::shared_ptr<UcxReceiverContext> rx_context) {
  gxf_result_t result = init_worker(ucp_context_, &rx_context->ucp_worker);
  if (result != GXF_SUCCESS) {
      return result;
  }

  result = register_am_recv_callback(rx_context->ucp_worker, rx_context);
  if (result != GXF_SUCCESS) {
      goto destroy_worker;
  }

  result = rx_context->rx->init_context(rx_context->ucp_worker,
          &rx_context->am_data_desc, efd_signal_, cpu_data_only_.get(),
          enable_async_.get());
  if (result != GXF_SUCCESS) {
      goto destroy_worker;
  }
  result = del_epoll_fd(epoll_fd_, rx_context->server_context.listener_fd);
  if (result != GXF_SUCCESS) {
      GXF_LOG_ERROR("failed to del listener fd %d from epoll",
              rx_context->server_context.listener_fd);
      goto deinit_context;
  }

  result = server_create_ep(rx_context);
  if (result != GXF_SUCCESS) {
      goto deinit_context;
  }

  rx_context->conn_state = CONNECTED;
  result = epoll_add_worker(rx_context, false);
  if (result != GXF_SUCCESS) {
      goto deinit_context;
  }
  rx_conns_.connected++;
  return GXF_SUCCESS;


deinit_context:
    rx_context->rx->init_context(NULL, NULL, 0, false, true);
destroy_worker:
    ucp_worker_destroy(rx_context->ucp_worker);
    return result;
}


gxf_result_t
UcxContext::am_desc_to_iov(std::shared_ptr<UcxReceiverContext> rx_context) {
  if (!cpu_data_only_.get()) {
    cudaError_t error{cudaSuccess};
    error = cudaSetDevice(dev_id_);
    if (error != cudaSuccess) {
        GXF_LOG_ERROR("cudaSetDevice Failed - %d", error);
        return GXF_FAILURE;
    }
  }
  rx_context->rx->buffer_.get()->reset();
  auto result = rx_context->rx->buffer_.get()->write(rx_context->am_data_desc.header,
                                                      rx_context->am_data_desc.header_length);
  if (!result) {
    if (result.error() == GXF_UNINITIALIZED_VALUE) {
      GXF_LOG_VERBOSE("Writing to SerializationBuffer failed. Will try again in 1ms.");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } else {
      GXF_LOG_ERROR("Writing to Serialization buffer failed with error %s",
                    GxfResultStr(result.error()));
    }
    return result.error();
  }
  free(rx_context->am_data_desc.header);
  auto message = entity_serializer_.get()->deserializeEntity(context(),
                                                              rx_context->rx->buffer_.get());
  if (!message) {
      GXF_LOG_ERROR("Deserialization failed with error %s", GxfResultStr(message.error()));
      return message.error();
  }
  rx_context->am_data_desc.num_of_comps = rx_context->rx->buffer_.get()->iov_buffer_size();

  ucp_dt_iov_t* recv_buf = const_cast<ucp_dt_iov_t*>(rx_context->rx->buffer_.get()->iov_buffer());
  rx_context->am_data_desc.recv_buf = rx_context->am_data_desc.num_of_comps == 1 ?
                                      recv_buf[0].buffer : recv_buf;
  rx_context->am_data_desc.msg_length = rx_context->am_data_desc.num_of_comps == 1 ?
                                        recv_buf[0].length :
                                        rx_context->am_data_desc.num_of_comps;
  rx_context->am_data_desc.mem_type = rx_context->rx->buffer_.get()->mem_type();
  rx_context->am_data_desc.receiving_message = true;
  if (enable_async_.get()) {
    rx_context->conn_state = RECEIVE_MESSAGE;
    if (del_epoll_fd(epoll_fd_, rx_context->worker_fd) != GXF_SUCCESS) {
        GXF_LOG_ERROR("failed to del fd %d from epoll for receiving message",
                rx_context->worker_fd);
        return GXF_FAILURE;
    }
    rx_context->rx->push(message.value());
  } else {
    rx_context->rx->push(message.value());
  }
  GxfEntityNotifyEventType(rx_context->rx->context(),
                            rx_context->rx->eid(),
                            GXF_EVENT_MESSAGE_SYNC);
  rx_context->am_data_desc.complete = 0;
  return GXF_SUCCESS;
}

void UcxContext::start_server_async_queue() {
  while (1) {
      if ((close_server_loop_) ||
              ((!reconnect_.get()) && (rx_conns_.closed == rx_conns_.total))) {
          break;
      }
      gxf_result_t result = wait_for_event();
      if (result != GXF_SUCCESS && result != GXF_UNINITIALIZED_VALUE) {
          GXF_LOG_ERROR("exit with error %s", GxfResultStr(result));
          return;
      }
  }
  GXF_LOG_DEBUG("Exit server loop");
  return;
}

void UcxContext::start_server() {
  GXF_LOG_INFO("UcxContext::start_server");
  bool all_conns_closed = false;
  while (1) {
      if (close_server_loop_) break;
      if (all_conns_closed) break;
      if (!reconnect_) all_conns_closed = true;
      for (auto rx_context_iter : rx_contexts_) {
          auto rx_context = rx_context_iter.value();
          switch (rx_context->conn_state) {
          // Close reset connection
          case RESET:
              rx_context->server_context.conn_request = NULL;
              rx_context->conn_state = CLOSED;
              // ep_close(rx_context->ucp_worker, rx_context->ep, 0);
              break;
          // Reconnect closed connection if reconnect_=true
          case CLOSED:
              rx_context->server_context.conn_request = NULL;
              if (!reconnect_)
                  break;
              rx_context->conn_state = INIT;
          // Create init connections
          case INIT:
              all_conns_closed = false;
              if (rx_context->server_context.conn_request == NULL) {
                  ucp_worker_progress(rx_context->server_context.listener_worker);
                  continue;
              }
              if (server_create_ep(rx_context) != GXF_SUCCESS) {
                  return;
              }
              rx_context->conn_state = CONNECTED;
              break;
          // Skip connected connections
          case CONNECTED:
              all_conns_closed = false;
              for (int i=0; i < WORKER_PROGRESS_RETRIES; i++) {
                 if (rx_context->am_data_desc.receiving_message) {
                      break;
                  }
                  if (!rx_context->am_data_desc.complete) {
                      ucp_worker_progress(rx_context->ucp_worker);
                  } else {
                      am_desc_to_iov(rx_context);
                      break;
                  }
              }
              break;
          default:
              break;
          }
      }
  }
  GXF_LOG_DEBUG("Exit server loop");
  return;
}

void UcxContext::copy_header_to_am_desc(std::shared_ptr<UcxReceiverContext> rx_context) {
  auto q_am_desc = rx_context->headers.front().value();
  ucx_am_data_desc* am_desc = static_cast<ucx_am_data_desc*>(&rx_context->am_data_desc);
  am_desc->header = malloc(q_am_desc->header_length);
  memcpy(am_desc->header, q_am_desc->header, q_am_desc->header_length);
  am_desc->header_length = q_am_desc->header_length;
  am_desc->msg_length = q_am_desc->msg_length;
  am_desc->desc    = q_am_desc->desc;
  am_desc->complete = q_am_desc->complete;
  rx_context->headers.erase(0);
}

gxf_result_t UcxContext::progress_work(std::shared_ptr<UcxReceiverContext> rx_context) {
  switch (rx_context->conn_state) {
  // Close reset connection
  case RESET:
      rx_conns_.closed++;
      rx_context->server_context.conn_request = NULL;
      if (reconnect_.get()) {
          rx_context->conn_state = INIT;
      } else {
          rx_context->conn_state = CLOSED;
      }
      break;
  // Reconnect closed connection if reconnect_=true
  case CLOSED:
      rx_context->server_context.conn_request = NULL;
      if (reconnect_.get()) {
        GXF_LOG_WARNING("Server endpoint connection was closed, reconnecting");
      } else {
        break;
      }
      rx_context->conn_state = INIT;
  // Create init connections
  case INIT:
      if (rx_context->server_context.conn_request == NULL) {
          while (ucp_worker_progress(rx_context->server_context.listener_worker)
                  && (rx_context->server_context.conn_request == NULL)) {}
      }
      if (rx_context->server_context.conn_request == NULL) {
          return GXF_SUCCESS;
      }
      if (init_connection(rx_context) != GXF_SUCCESS) {
          GXF_LOG_ERROR("failed to init_connection");
          return GXF_FAILURE;
      }
      break;
  // Skip connected connections
  case CONNECTED:
      if (!rx_context->am_data_desc.receiving_message) {
          if ((rx_context->headers.size() > 0) && (!rx_context->am_data_desc.complete)) {
              copy_header_to_am_desc(rx_context);
          }
          if (!rx_context->am_data_desc.complete) {
              int res_worker;
              do {
                  res_worker = ucp_worker_progress(rx_context->ucp_worker);
              } while (res_worker && (!rx_context->am_data_desc.complete));
          }
          if (rx_context->am_data_desc.complete) {
              gxf_result_t result = am_desc_to_iov(rx_context);
              if (result != GXF_SUCCESS) {
                  if (result != GXF_UNINITIALIZED_VALUE) {
                      GXF_LOG_ERROR("failed to process header for received message with error %s",
                                    GxfResultStr(result));
                  }
                  return result;
              }
              break;
          }
      }
      break;
  default:
      break;
  }
  return GXF_SUCCESS;
}



gxf_result_t UcxContext::handle_connections_after_recv() {
  uint64_t val;
  // need to read fd to reset in order to be able rerun epoll on it
  if ((read(efd_signal_, &val, sizeof(val))) == -1) {
      GXF_LOG_ERROR("failed to reset signaling fd");
  }
  for (auto rx_context_iter : rx_contexts_) {
      auto rx_context = rx_context_iter.value();
      if ((!rx_context->am_data_desc.receiving_message) &&
              (rx_context->conn_state == RECEIVE_MESSAGE)) {
          rx_context->conn_state = CONNECTED;
          if (add_epoll_fd(epoll_fd_, rx_context->worker_fd) != GXF_SUCCESS) {
              GXF_LOG_ERROR("failed to add fd %d to epoll", epoll_fd_);
              return GXF_FAILURE;
          }
          gxf_result_t result = progress_work(rx_context);
          if (result != GXF_SUCCESS) {
              if (result != GXF_UNINITIALIZED_VALUE) {
                  GXF_LOG_ERROR("failed to progress worker with error %s", GxfResultStr(result));
              }
              return result;
          }
      }
  }
  return GXF_SUCCESS;
}

gxf_result_t UcxContext::wait_for_event() {
  int num_events = 0;
  size_t num_rx_contexts = rx_contexts_.size();
  std::vector<epoll_event> events(num_rx_contexts);
  do {
      num_events = epoll_wait(epoll_fd_, events.data(), num_rx_contexts, -1);
  } while ((num_events == -1) || (errno == EINTR));

  for (int i = 0; i < num_events; i++) {
      if (events[i].data.fd == efd_signal_) {
          gxf_result_t result = handle_connections_after_recv();
          if (result != GXF_SUCCESS) {
              if (result != GXF_UNINITIALIZED_VALUE) {
                  GXF_LOG_ERROR("failed to handle connection after received message with error %s",
                                GxfResultStr(result));
              }
              return result;
          }
          continue;
      }
      int found_worker = false;
      for (auto rx_context_iter : rx_contexts_) {
          auto rx_context = rx_context_iter.value();
          if ((rx_context->worker_fd == events[i].data.fd) ||
                  (rx_context->server_context.listener_fd == events[i].data.fd)) {
              found_worker = true;
              gxf_result_t result = progress_work(rx_context);
              if (result != GXF_SUCCESS) {
                  if (result != GXF_UNINITIALIZED_VALUE) {
                      GXF_LOG_ERROR("failed to progress worker with error %s",
                                    GxfResultStr(result));
                  }
                  return result;
              }
              if (!rx_context->am_data_desc.receiving_message) {
                  auto result = gxf_arm_worker(rx_context,
                                                rx_context->conn_state == INIT);
                  if (result !=  GXF_SUCCESS) {
                      if (result != GXF_UNINITIALIZED_VALUE) {
                          GXF_LOG_ERROR("failed to arm fd %d to epoll with error %s",
                                        rx_context->worker_fd, GxfResultStr(result));
                      }
                      return result;
                  }
              }
              break;
          }
      }
      if (!found_worker) {
          GXF_LOG_ERROR("worker_fd not found in UcxContext");
          return GXF_FAILURE;
      }
  }
  return GXF_SUCCESS;
}

/**
 * Initialize the UCP context and worker.
 */

gxf_result_t UcxContext::init_context() {
  ucp_params_t ucp_params;
  ucs_status_t status;

  memset(&ucp_params, 0, sizeof(ucp_params));
  ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_NAME;
  ucp_params.name       = "client_server";
  ucp_params.features = UCP_FEATURE_AM;

  if (enable_async_.get() && epoll_fd_ != -1) {
      ucp_params.features |= UCP_FEATURE_WAKEUP;
  }
  ucp_params.field_mask |= UCP_PARAM_FIELD_MT_WORKERS_SHARED;
  ucp_params.mt_workers_shared = 1;

  status = ucp_init(&ucp_params, NULL, &ucp_context_);
  if (status != UCS_OK) {
      GXF_LOG_ERROR("failed to ucp_init (%s)", ucs_status_string(status));
      return GXF_FAILURE;
  }
  if (enable_async_.get()) {
    close_server_loop_ = false;
    epoll_fd_ = epoll_create(1);
    if (epoll_fd_ < 0) {
      GXF_LOG_ERROR("failed to create epoll fd");
      return GXF_FAILURE;
    }
    efd_signal_ = eventfd(0, 0);
    if (add_epoll_fd(epoll_fd_, efd_signal_) != GXF_SUCCESS) {
      GXF_LOG_ERROR("failed to add signal fd (%d) to epoll", efd_signal_);
      return GXF_FAILURE;
    }
  }
  return GXF_SUCCESS;
}


/**
 * EPOLL functions
 */
gxf_result_t UcxContext::gxf_arm_worker(
        std::shared_ptr<UcxReceiverContext> rx_context, bool is_listener) {
  ucs_status_t status;
  ucp_worker_h worker = is_listener ?
          rx_context->server_context.listener_worker : rx_context->ucp_worker;
  do {
    auto result = progress_work(rx_context);
    if (result != GXF_SUCCESS) {
        if (result != GXF_UNINITIALIZED_VALUE) {
            GXF_LOG_ERROR("failed to progress worker with error %s", GxfResultStr(result));
        }
        return result;
    }
    status = ucp_worker_arm(worker);
  } while (status == UCS_ERR_BUSY);
  if (status != UCS_OK) {
    GXF_LOG_ERROR("ucp_epoll error: %s", ucs_status_string(status));
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

static gxf_result_t add_epoll_fd(int epoll_fd, int fd) {
  epoll_event ev;
  memset(&ev, 0, sizeof(ev));
  ev.events  = EPOLLIN;
  ev.data.fd = fd;
  if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &ev) == -1) {
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

static gxf_result_t del_epoll_fd(int epoll_fd, int fd) {
  if (epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, NULL) == -1) {
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}


}  // namespace gxf
}  // namespace nvidia
