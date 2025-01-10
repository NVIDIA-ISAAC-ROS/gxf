/*
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/ipc/http/tests/mock_http_service.hpp"

#include <string>

namespace nvidia {
namespace gxf {

static constexpr const char* kTestQueryService = "test_query_service";
static constexpr const char* kTestActionService = "test_action_service";
static constexpr const char* kTestResource = "menu";
static constexpr const char* kTestActionPayload = "{\"key1\": \"value1\", \"key2\": \"value2\"}";
static constexpr const char* kTestNonExistService = "non_exist_service";

// Mock service Implementation
gxf_result_t MockHttpService::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(server_, "server", "API server",
                                 "API Server for remote access to the realtime statistic data",
                                 Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  return ToResultCode(result);
}

gxf_result_t MockHttpService::initialize() {
  auto maybe_server = server_.try_get();
  if (maybe_server) {
    auto server_handle = maybe_server.value();
    // register sample query service, HTTP GET
    IPCServer::Service service_sample_query = {
      kTestQueryService,
      IPCServer::kQuery,
      {.query = std::bind(&MockHttpService::onMockQuery, this, std::placeholders::_1)}
    };
    server_handle->registerService(service_sample_query);

    // register sample action service, eg HTTP POST
    IPCServer::Service service_sample_action = {
      kTestActionService,
      IPCServer::kAction,
      {.action = std::bind(&MockHttpService::onMockAction, this, std::placeholders::_1,
                           std::placeholders::_2)}
    };
    server_handle->registerService(service_sample_action);
  }
  return GXF_SUCCESS;
}

gxf_result_t MockHttpService::deinitialize() {
  return GXF_SUCCESS;
}

Expected<std::string> MockHttpService::onMockQuery(const std::string& resource) {
  // User service logic goes here
  GXF_LOG_INFO("Service querying on resource: %s", resource.c_str());
  return "{\"key1\": \"value1_mock\", \"key2\": \"value2_mock\"}";
}

Expected<void> MockHttpService::onMockAction(const std::string& resource,
                                             const std::string& data) {
  // User service logic goes here
  Expected<void> result;
  GXF_LOG_INFO("Service taking action on resource: %s, with data: %s\n", resource.c_str(),
               data.c_str());
  return result;
}


// Mock Client Implementation, legacy
gxf_result_t MockHttpClient::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(client_, "client", "HTTP Client",
                                 "Mock HTTP Client",
                                 Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  return ToResultCode(result);
}

gxf_result_t MockHttpClient::tick() {
  auto maybe_client = client_.try_get();
  if (maybe_client) {
    auto client_handle = maybe_client.value();
    {
      std::string uri_query = "test_query_service/menu";  // path/resource
      Expected<nvidia::gxf::HttpClient::Response> response
        = client_handle->getRequest(uri_query);
      if (!response) {
        return response.error();
      }
      int status_code = response.value().status_code;
      GXF_ASSERT(status_code == 200, "Error Http GET status code %d", status_code);
      std::stringstream ss;
      ss << "Good GET status code: " << status_code;
      if (!response.value().body.empty()) {
        ss << ", response body: " + response.value().body;
      }
      ss << std::endl;
      GXF_LOG_INFO("%s", ss.str().c_str());
    }

    {
      std::string uri_action = "test_action_service/menu";  // path/resource
      std::string data_action = "{\"key1\": \"value1\", \"key2\": \"value2\"}";
      std::string content_type = "application/json";
      Expected<nvidia::gxf::HttpClient::Response> response
        = client_handle->postRequest(uri_action, data_action, content_type);
      if (!response) {
        return response.error();
      }
      int status_code = response.value().status_code;
      GXF_ASSERT((status_code == 200 || status_code == 201 || status_code == 202),
                  "Error Http POST status code %d", status_code);
      std::stringstream ss;
      ss << "Good POST status code: " << status_code;
      if (!response.value().body.empty()) {
        ss << ", response body: " + response.value().body;
      }
      ss << std::endl;
      GXF_LOG_INFO("%s", ss.str().c_str());
    }
  }
  return GXF_SUCCESS;
}

// Mock Client Implementation
gxf_result_t MockHttpIPCClient::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(client_, "client", "HTTP Client",
                                 "Mock HTTP Client",
                                 Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  return ToResultCode(result);
}

gxf_result_t MockHttpIPCClient::tick() {
  auto maybe_client = client_.try_get();
  if (maybe_client) {
    auto client_handle = maybe_client.value();
    {
      Expected<std::string> response = client_handle->query(kTestQueryService, kTestResource);
      if (!response) {
        return response.error();
      }
      GXF_LOG_INFO("Successfully got query response: %s from server 1", response.value().c_str());
    }

    {
      Expected<void> response = client_handle->action(
        kTestActionService, kTestResource, kTestActionPayload);
      if (!response) {
        return response.error();
      }
      GXF_LOG_INFO("Successfully sent action request to server 1");
    }

    {
      Expected<std::string> response = client_handle->ping();
      if (!response) {
        return response.error();
      }
      GXF_LOG_INFO("\n\nSuccessfully got ping response: [%s] from server 1 / default ping service",
        response.value().c_str());
    }

    {
      Expected<std::string> response = client_handle->changeAddress("localhost", 8083).ping();
      if (!response) {
        return response.error();
      }
      GXF_LOG_INFO("\n\nSuccessfully got ping response: [%s] from server 2 / default ping service",
        response.value().c_str());
    }
  }
  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
