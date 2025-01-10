/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/ipc/http/http_ipc_client.hpp"

#include "cpprest/http_client.h"  // NOLINT

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nvidia {
namespace gxf {

class HttpIPCClient::Impl {
 public:
  std::unique_ptr<web::uri> base_uri_;
  std::unique_ptr<web::http::client::http_client> raw_client_;
};

void HttpIPCClient::ImplDeleter::operator()(Impl* ptr) {
  delete ptr;
}

gxf_result_t HttpIPCClient::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      port_, "port", "GRPC port for connecting", "HTTP port for connecting", 50000U);
  result &= registrar->parameter(
      server_ip_address_, "server_ip_address",
      "Server IP address", "Server IP address", std::string("0.0.0.0"));
  result &= registrar->parameter(
      use_https_, "use_https", "use Https",
      "Use TLS(SSL). If true, protocol is https. Otherwise protocol is http.", false);
  result &= registrar->parameter(
      content_type_, "content-type",
      "Http content_type", "Http content_type", std::string("application/json"));
  return ToResultCode(result);
}

gxf_result_t HttpIPCClient::initialize() {
  try {
    client_ = std::unique_ptr<Impl, ImplDeleter>(new Impl);
    utility::string_t string_uri;
    if (use_https_.get()) {
      string_uri = "https";
    } else {
      string_uri = "http";
    }
    string_uri += "://" + this->toIpPort(server_ip_address_.get(), port_.get()) + "/";
    client_->base_uri_ = std::make_unique<web::uri>(string_uri);
    client_->raw_client_ = std::make_unique<web::http::client::http_client>(
      *(client_->base_uri_.get()));
    GXF_LOG_INFO("Initialize HTTP client base_uri: %s",
                 client_->raw_client_->base_uri().to_string().c_str());
  } catch (std::exception &e) {
    GXF_LOG_ERROR("Exception happens while HTTP client is initializing: %s", e.what());
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t HttpIPCClient::deinitialize() {
  client_->base_uri_.reset(nullptr);
  client_->raw_client_.reset(nullptr);
  return GXF_SUCCESS;
}

HttpIPCClient& HttpIPCClient::changeAddress(const std::string& ip, uint32_t port) {
  client_->base_uri_.reset(nullptr);
  client_->raw_client_.reset(nullptr);
  try {
    utility::string_t string_uri;
    if (use_https_.get()) {
      string_uri = "https";
    } else {
      string_uri = "http";
    }
    string_uri += "://" + this->toIpPort(ip, port) + "/";
    client_->base_uri_ = std::make_unique<web::uri>(string_uri);
    client_->raw_client_ = std::make_unique<web::http::client::http_client>(
      *(client_->base_uri_.get()));
    GXF_LOG_DEBUG("Initialize HTTP client base_uri: %s",
                 client_->raw_client_->base_uri().to_string().c_str());
  } catch (std::exception &e) {
    GXF_LOG_ERROR("Exception happens while HTTP client change server address: %s", e.what());
  }
  return *this;
}

Expected<std::string> HttpIPCClient::query(
  const std::string& service,
  const std::string& resource
) {
  if (client_ == nullptr || client_->raw_client_ == nullptr) {
    GXF_LOG_ERROR("HttpIPCClient invalid raw client");
    return Unexpected{GXF_PARAMETER_MANDATORY_NOT_SET};
  }
  if (service.empty() || resource.empty()) {
    GXF_LOG_ERROR("Invalid arguments in IPCClient query call");
    return Unexpected{GXF_ARGUMENT_INVALID};
  }

  Response response_get;
  const std::string uri = service + "/" + resource;
  // Make a GET request
  auto task = client_->raw_client_->request(web::http::methods::GET, uri)
  // Get the response
  .then([&response_get](web::http::http_response response) {
    // Check the status code.
    if (response.status_code() != 200) {
      GXF_LOG_ERROR("Status code: %s", std::to_string(response.status_code()).c_str());
    }
    response_get.status_code = response.status_code();
    // extract the response body to string, async.
    return response.extract_string();
  })
  // Get response body as string
  .then([&response_get](utility::string_t respose_body) {
    response_get.body = respose_body;
  });
  // Wait for the concurrent tasks to finish.
  try {
    task.wait();
  } catch (const std::exception &e) {
    GXF_LOG_ERROR("Exception:%s\n", e.what());
    return Unexpected{GXF_HTTP_GET_FAILURE};
  }

  return response_get.body;  // auto move
}

Expected<void> HttpIPCClient::action(
  const std::string& service,
  const std::string& resource,
  const std::string& data
) {
  if (client_ == nullptr || client_->raw_client_ == nullptr) {
    GXF_LOG_ERROR("HttpIPCClient invalid raw client");
    return Unexpected{GXF_PARAMETER_MANDATORY_NOT_SET};
  }
  if (service.empty() || resource.empty()) {
    GXF_LOG_ERROR("Invalid arguments in IPCClient query call");
    return Unexpected{GXF_ARGUMENT_INVALID};
  }

  const std::string uri = service + "/" + resource;
  Response response_post;
  // Make a POST request
  auto task = client_->raw_client_->request(
    web::http::methods::POST, uri, data, content_type_.get()
  )
  // Get the response.
  .then([&response_post](web::http::http_response response) {
    // Check the status code.
    if (response.status_code() != 200 &&
        response.status_code() != 201 &&
        response.status_code() != 202) {
      GXF_LOG_ERROR("Status code: %s", std::to_string(response.status_code()).c_str());
    }
    response_post.status_code = response.status_code();
    // extract the response body to string.
    return response.extract_string();
  })
  // Get response body as string
  .then([&response_post](utility::string_t respose_body) {
    response_post.body = respose_body;
  });
  // Wait for the concurrent tasks to finish.
  try {
    task.wait();
  } catch (const std::exception &e) {
    GXF_LOG_ERROR("Exception:%s\n", e.what());
    return Unexpected{GXF_HTTP_POST_FAILURE};
  }

  return Success;
}

Expected<std::string> HttpIPCClient::ping(const std::string& service) {
  if (service == kDefaultPingServiceName) {  // IPCServer's own generic ping service
    return this->query(kDefaultPingServiceName, "gxf::IPCServer");
  } else {  // Http Health Check Protocol impl, if any
    return Unexpected{GXF_NOT_IMPLEMENTED};
  }
}

}  // namespace gxf
}  // namespace nvidia
