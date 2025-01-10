/*
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/ipc/http/http_server.hpp"

#include <memory>
#include <string>

namespace nvidia {
namespace gxf {

gxf_result_t HttpServer::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(port_, "port", "HTTP port for listening",
                                 "HTTP port for listening", 8000U);
  result &= registrar->parameter(remote_access_, "remote_access",
                                 "Allow access from a remote client",
                                 "Flag to control remote access", false);
  return ToResultCode(result);
}

gxf_result_t HttpServer::initialize() {
  // define the endpoint
  try {
    std::string endpoint = remote_access_ ? "http://0.0.0.0:":"http://127.0.0.1:";
    endpoint += std::to_string(port_);
    endpoint += "/";
    // construct the listener
    listener_ = std::make_unique<listener>(endpoint);
    // declare the HTTP request support
    listener_->support(std::bind(&HttpServer::handleRequest, this, std::placeholders::_1));
    // Ping handler
    std::string path = "ping";
    IPCServer::QueryHandler handler = std::bind(&HttpServer::onPing, this, std::placeholders::_1);
    installHandler(get_handlers_, path, handler);
    // start the listener
    listener_->open().wait();
  } catch (std::exception &e) {
    GXF_LOG_ERROR("Exception happens while HTTP listener is initializing: %s", e.what());
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t HttpServer::deinitialize() {
  listener_->close().wait();
  return GXF_SUCCESS;
}
void HttpServer::handleRequest(web::http::http_request request) {
  GXF_LOG_VERBOSE("Handling HTTP request: '%s'", request.to_string().c_str());

  auto params = uri::split_path(uri::decode(request.relative_uri().path()));
  if (params.size() < 2) {
    request.reply(status_codes::BadRequest);
    return;
  }
  // the first section of the URL indicates the name of the requested service
  auto& path = params[0];
  // all the left indicates the resource to be accessed
  auto& resource = params[1];
  for (unsigned int i = 2; i < params.size(); i++) {
    resource = resource + "/" + params[i];
  }

  std::unique_lock<std::mutex> lock(mutex_);
  // handling GET request
  if (
    request.method() == methods::GET &&
    get_handlers_.find(path) != get_handlers_.end()
  ) {
    auto response = get_handlers_[path](resource);
    if (response) {
      request.reply(status_codes::OK, response.value());
    } else {
      request.reply(status_codes::InternalError, std::string());
    }
  } else if (
    request.method() == methods::POST &&
    post_handlers_.find(path) != post_handlers_.end()
  ) {
    auto data = request.extract_string().get();
    if (!data.empty()) {
      auto response = post_handlers_[path](resource, data);
      if (response) {
        request.reply(status_codes::OK, std::string());
      } else {
        request.reply(status_codes::InternalError, "Failure");
      }
    } else {
      request.reply(status_codes::BadRequest);
    }
  } else {
    request.reply(status_codes::BadRequest);
  }
}

}  // namespace gxf
}  // namespace nvidia
