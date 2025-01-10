/*
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/ipc/grpc/tests/mock_ipc_service_user.hpp"

#include <string>
#include "gxf/ipc/grpc/grpc_server.hpp"

namespace nvidia {
namespace gxf {

static constexpr const char* kTestQueryService = "test_query_service";
static constexpr const char* kTestActionService = "test_action_service";
static constexpr const char* kTestResource = "menu";
static constexpr const char* kTestActionPayload = "{\"key1\": \"value1\", \"key2\": \"value2\"}";
static constexpr const char* kTestNonExistService = "non_exist_service";

// Mock service Implementation
gxf_result_t MockIPCServiceUser::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(server_, "server", "API server",
                                 "API Server for remote access to the realtime statistic data",
                                 Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  return ToResultCode(result);
}

gxf_result_t MockIPCServiceUser::initialize() {
  auto maybe_server = server_.try_get();
  if (maybe_server) {
    auto server_handle = maybe_server.value();
    // register sample query service
    IPCServer::Service service_sample_query = {
      kTestQueryService,
      IPCServer::kQuery,
      {.query = std::bind(&MockIPCServiceUser::onMockQuery, this, std::placeholders::_1)}
    };
    server_handle->registerService(service_sample_query);

    // register sample action service
    IPCServer::Service service_sample_action = {
      kTestActionService,
      IPCServer::kAction,
      {.action = std::bind(&MockIPCServiceUser::onMockAction, this, std::placeholders::_1,
                           std::placeholders::_2)}
    };
    server_handle->registerService(service_sample_action);
  }
  return GXF_SUCCESS;
}

gxf_result_t MockIPCServiceUser::deinitialize() {
  return GXF_SUCCESS;
}

Expected<std::string> MockIPCServiceUser::onMockQuery(const std::string& resource) {
  // User service logic goes here
  GXF_LOG_INFO("Service querying on resource: %s", resource.c_str());
  return "{\"key1\": \"value1_mock\", \"key2\": \"value2_mock\"}";
}

Expected<void> MockIPCServiceUser::onMockAction(const std::string& resource,
                                             const std::string& data) {
  // User service logic goes here
  Expected<void> result;
  GXF_LOG_INFO("Service taking action on resource: %s, with data: %s\n", resource.c_str(),
               data.c_str());
  return result;
}


// Mock Client Implementation
gxf_result_t MockIPCClientUser::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(client_, "client", "GRPC Client",
                                 "Mock GRPC Client",
                                 Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  return ToResultCode(result);
}

gxf_result_t MockIPCClientUser::tick() {
  auto maybe_client = client_.try_get();
  if (maybe_client) {
    auto client_handle = maybe_client.value();
    /**
     * Normal tests
    */
    {  // query() test
      Expected<std::string> response = client_handle->query(kTestQueryService, kTestResource);
      if (!response) {
        return response.error();
      }
      GXF_LOG_INFO("Successfully got query response: %s from server 1", response.value().c_str());
    }
    {  // action() test
      Expected<void> response = client_handle->action(
        kTestActionService, kTestResource, kTestActionPayload);
      if (!response) {
        return response.error();
      }
      GXF_LOG_INFO("Successfully sent action request to server 1");
    }

    /**
     * Ping tests
    */
    {  // IPCServer defatul ping() test
      Expected<std::string> response = client_handle->ping("ping");
      if (!response) {
        return response.error();
      }
      GXF_LOG_INFO("\n\nSuccessfully got ping response: [%s] from server 1 / default ping service",
        response.value().c_str());
    }
    {  // gRPC Health Check Protocol, gxf::IPCServer main entrance service test
      Expected<std::string> response = client_handle->ping(GrpcServer::kGrpcMainEntranceService);
      if (!response) {
        return response.error();
      }
      GXF_LOG_INFO("Successfully got ping response: [%s] from server 1 / service[%s]",
        response.value().c_str(), GrpcServer::kGrpcMainEntranceService);
    }
    {  // gRPC Health Check Protocol, gxf::IPCServer user registered service test
      Expected<std::string> response = client_handle->ping(kTestQueryService);
      if (!response) {
        return response.error();
      }
      GXF_LOG_INFO("Successfully got ping response: [%s] from server 1 / service[%s]",
        response.value().c_str(), kTestQueryService);
    }
    {  // gRPC Health Check Protocol, gxf::IPCServer user registered service test
      Expected<std::string> response = client_handle->ping(kTestActionService);
      if (!response) {
        return response.error();
      }
      GXF_LOG_INFO("Successfully got ping response: [%s] from server 1 / service[%s]",
        response.value().c_str(), kTestActionService);
    }
    {  // For non_exist_service, first health check then query
      Expected<std::string> response2 = client_handle->ping(kTestNonExistService);
      if (!response2) {
        if (response2.error() != GXF_IPC_SERVICE_NOT_FOUND) {
          return response2.error();
        }
        GXF_LOG_INFO("GXF_LOG_ERROR expected for negative test: ping non_exist_service");
      }
      Expected<std::string> response = client_handle->query(kTestNonExistService, kTestResource);
      if (!response) {
        if (response.error() != GXF_IPC_SERVICE_NOT_FOUND) {
          return response.error();
        }
        GXF_LOG_INFO("GXF_LOG_ERROR expected for negative test: call non_exist_service");
      }
    }
    {  // For non_exist_service, health check after query
      Expected<std::string> response2 = client_handle->ping(kTestNonExistService);
      if (!response2) {
        return response2.error();
      }
      GXF_LOG_INFO("Successfully got ping response: [%s] from server 1 / service[%s]",
        response2.value().c_str(), kTestNonExistService);
    }

    /**
     * Optional inline changeAddress tests
    */
    {  // inline changeAddress query() test
      Expected<std::string> response =
        client_handle->changeAddress("localhost", 50001).query(kTestQueryService, kTestResource);
      if (!response) {
        return response.error();
      }
      GXF_LOG_INFO("\n\nSuccessfully got query response: %s from server 2",
        response.value().c_str());
    }
    {  // inline changeAddress action() test
      Expected<void> response =
        client_handle->changeAddress("localhost", 50001).action(kTestActionService,
          kTestResource, kTestActionPayload);
      if (!response) {
        return response.error();
      }
      GXF_LOG_INFO("Successfully sent action request to server 2");
    }
  }
  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
