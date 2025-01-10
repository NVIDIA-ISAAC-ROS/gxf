/*
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/ipc/http/http_client_cpprest_impl.hpp"

#include <memory>
#include <string>

namespace nvidia {
namespace gxf {

gxf_result_t CppRestHttpClient::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      server_ip_port_, "server_ip_port", "server ip port",
      "Server IP and Port.", std::string());
  result &= registrar->parameter(
      use_https_, "use_https", "use Https",
      "Use TLS(SSL). If true, protocol is https. Otherwise protocol is http.", false);
  return ToResultCode(result);
}

gxf_result_t CppRestHttpClient::initialize() {
  try {
    utility::string_t string_uri;
    if (use_https_.get()) {
      string_uri = "https";
    } else {
      string_uri = "http";
    }
    string_uri += "://" + server_ip_port_.get() + "/";
    base_uri_ = std::make_unique<web::uri>(string_uri);
    // base_uri_ = std::make_unique<web::uri>(U"http://localhost:8080");
    raw_client_ = std::make_unique<web::http::client::http_client>(*(base_uri_.get()));
    // raw_client_ = std::make_unique<web::http::client::http_client>(string_uri);
    GXF_LOG_INFO("Initialize HTTP client base_uri: %s",
                 raw_client_->base_uri().to_string().c_str());
  } catch (std::exception &e) {
    GXF_LOG_ERROR("Exception happens while HTTP client is initializing: %s", e.what());
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t CppRestHttpClient::deinitialize() {
  return GXF_SUCCESS;
}

Expected<nvidia::gxf::HttpClient::Response>
CppRestHttpClient::getRequest(const std::string& uri) {
  nvidia::gxf::HttpClient::Response response_get;
  // Make a GET request
  auto task = raw_client_->request(web::http::methods::GET, uri)

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

  return response_get;  // reply on auto move
}

Expected<nvidia::gxf::HttpClient::Response>
CppRestHttpClient::postRequest(const std::string& uri,
                                   const std::string& payload, const std::string& content_type) {
  nvidia::gxf::HttpClient::Response response_post;
  // Make a POST request
  auto task = raw_client_->request(web::http::methods::POST, uri, payload, content_type)

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

  return response_post;  // rely on auto move
}

}  // namespace gxf
}  // namespace nvidia
