/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <arpa/inet.h> /* inet_addr */
#include <ucp/api/ucp.h>
#include <cstdlib>
#include <cstring>
#include <string>

#include "ucx_common.hpp"


namespace nvidia {
namespace gxf {

static sa_family_t ai_family   = AF_INET;

/**
 * Set an address for the server to listen on - INADDR_ANY on a well known port.
 */
void set_sock_addr(const char* address_str, int port, struct sockaddr_storage* saddr) {
    struct sockaddr_in *sa_in;
    struct sockaddr_in6 *sa_in6;

    /* The server will listen on INADDR_ANY */
    memset(saddr, 0, sizeof(*saddr));

    switch (ai_family) {
    case AF_INET:
        sa_in = (struct sockaddr_in*)saddr;
        if ((address_str != NULL) && (strcmp(address_str, "0.0.0.0"))) {
            inet_pton(AF_INET, address_str, &sa_in->sin_addr);
        } else {
            sa_in->sin_addr.s_addr = INADDR_ANY;
        }
        sa_in->sin_family = AF_INET;
        sa_in->sin_port   = htons(port);
        break;
    case AF_INET6:
        sa_in6 = (struct sockaddr_in6*)saddr;
        if ((address_str != NULL) && (strcmp(address_str, "0.0.0.0"))) {
            inet_pton(AF_INET6, address_str, &sa_in6->sin6_addr);
        } else {
            sa_in6->sin6_addr = in6addr_any;
        }
        sa_in6->sin6_family = AF_INET6;
        sa_in6->sin6_port   = htons(port);
        break;
    default:
        GXF_LOG_ERROR("Invalid address family");
        break;
    }
}

gxf_result_t request_finalize(ucp_worker_h ucp_worker, void* request,
                                           test_req_t* ctx) {
    ucs_status_t status;
    /* if operation was completed immediately */
    if (request == NULL) {
        return GXF_SUCCESS;
    }

    if (UCS_PTR_IS_ERR(request)) {
        GXF_LOG_ERROR("Unable to handle UCX message (%s)",
                      ucs_status_string(UCS_PTR_STATUS(request)));
        return GXF_FAILURE;
    }

    for (int i = 0; (i < WORKER_PROGRESS_ITERATIONS) && (ctx->complete == 0); i++) {
        ucp_worker_progress(ucp_worker);
    }

    if (ctx->complete != 0) {
        status = ucp_request_check_status(request);
        ucp_request_free(request);
        if (ctx->header)
            free(ctx->header);
        free(ctx);
        if (status != UCS_OK) {
            GXF_LOG_ERROR("Unable to receive UCX message (%s)",
                          ucs_status_string(UCS_PTR_STATUS(request)));
            return GXF_FAILURE;
        }
        return GXF_SUCCESS;
    }
    return GXF_NOT_FINISHED;
}

ucs_status_t request_wait(ucp_worker_h ucp_worker, void* request,
                          test_req_t* ctx) {
    ucs_status_t status;

    /* if operation was completed immediately */
    if (request == NULL) {
        return UCS_OK;
    }

    if (UCS_PTR_IS_ERR(request)) {
        return UCS_PTR_STATUS(request);
    }

    while (ctx->complete == 0) {
        ucp_worker_progress(ucp_worker);
    }
    status = ucp_request_check_status(request);

    ucp_request_free(request);
    free(ctx);
    return status;
}

ucs_status_t request_wait_once(ucp_worker_h ucp_worker, void* request,
                                 test_req_t* ctx) {
    ucs_status_t status;

    /* if operation was completed immediately */
    if (request == NULL) {
        return UCS_OK;
    }

    if (UCS_PTR_IS_ERR(request)) {
        return UCS_PTR_STATUS(request);
    }

    // while (ctx->complete == 0) {
        ucp_worker_progress(ucp_worker);
    // }
    status = ucp_request_check_status(request);

    // ucp_request_free(request);

    return status;
}

ucs_status_t process_request(ucp_worker_h ucp_worker, void* req) {
    ucs_status_t status;

    if (req == NULL) {
        return UCS_OK;
    }
    if (UCS_PTR_IS_PTR(req)) {
        do {
            ucp_worker_progress(ucp_worker);
            status = ucp_request_check_status(req);
        } while (status == UCS_INPROGRESS);
        ucp_request_free(req);
    } else {
        status = UCS_PTR_STATUS(req);
    }
    return status;
}

void ep_close(ucp_worker_h ucp_worker, ucp_ep_h ep, uint32_t flags) {
    ucp_request_param_t param;
    ucs_status_t status;
    void *close_req;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    param.flags        = flags;
    param.flags        = UCP_EP_CLOSE_FLAG_FORCE;
    close_req          = ucp_ep_close_nbx(ep, &param);
    status = process_request(ucp_worker, close_req);
    if (status != UCS_OK) {
        GXF_LOG_ERROR("failed to close ep %p:%d %s ", (void*)ep, status,
                ucs_status_string(status));
    }
}

/**
 * Create a ucp worker on the given ucp context.
 */
gxf_result_t init_worker(ucp_context_h ucp_context, ucp_worker_h* ucp_worker) {
    ucp_worker_params_t worker_params;
    ucs_status_t status;

    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;

    status = ucp_worker_create(ucp_context, &worker_params, ucp_worker);
    if (status != UCS_OK) {
        GXF_LOG_ERROR("failed to ucp_worker_create (%s)", ucs_status_string(status));
        return GXF_FAILURE;
    }
    return GXF_SUCCESS;
}

// Return UCX Memory type based on GXF mem type
ucs_memory_type_t ucx_mem_type(MemoryStorageType gxf_mem_type) {
    switch (gxf_mem_type) {
    case MemoryStorageType::kHost:
        return ucs_memory_type::UCS_MEMORY_TYPE_CUDA_MANAGED;
    case MemoryStorageType::kDevice:
        return ucs_memory_type::UCS_MEMORY_TYPE_CUDA;
    case MemoryStorageType::kSystem:
        return ucs_memory_type::UCS_MEMORY_TYPE_HOST;
    }
    return ucs_memory_type::UCS_MEMORY_TYPE_HOST;
}


}  // namespace gxf
}  // namespace nvidia
