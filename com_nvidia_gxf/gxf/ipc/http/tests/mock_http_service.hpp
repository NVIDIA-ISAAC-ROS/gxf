/*
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_HTTP_TESTS_MOCK_HTTP_SERVICE_HPP_
#define NVIDIA_GXF_HTTP_TESTS_MOCK_HTTP_SERVICE_HPP_

#include <cstdint>
#include <string>

#include "gxf/core/component.hpp"
#include "gxf/ipc/http/http_client.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/ipc_client.hpp"
#include "gxf/std/ipc_server.hpp"

namespace nvidia {
namespace gxf {

/// @brief Mock Http service based on IPCServer
class MockHttpService : public Component {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override;

 private:
  // API Server for remote access
  Parameter<Handle<IPCServer>> server_;

  Expected<std::string> onMockQuery(const std::string& resource);
  Expected<void> onMockAction(const std::string& resource, const std::string& data);
};

/// @brief Mock Http client based on HttpClient
class MockHttpClient : public Codelet {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t tick() override;

 private:
  // client for remote access
  Parameter<Handle<HttpClient>> client_;
};

/// @brief Mock Http client based on HttpIPCClient
class MockHttpIPCClient : public Codelet {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t tick() override;

 private:
  // client for remote access
  Parameter<Handle<IPCClient>> client_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_HTTP_TESTS_MOCK_HTTP_SERVICE_HPP_
