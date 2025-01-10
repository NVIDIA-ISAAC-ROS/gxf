/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/ipc/grpc/grpc_client.hpp"

#include <grpcpp/grpcpp.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gxf/ipc/grpc/grpc_service.grpc.pb.h"

// The gRPC installation includes the health module only for Bazel build systems.
// Other build systems (CMake) must install and include the health module manually.
#ifdef CMAKE_BUILD
#include "health.grpc.pb.h"
#else
#include "src/proto/grpc/health/v1/health.grpc.pb.h"
#endif

namespace nvidia {
namespace gxf {

class GrpcClient::Impl {
 public:
  std::unique_ptr<::gxf::ServiceHub::Stub> stub_;
  std::unique_ptr<grpc::health::v1::Health::Stub> health_stub_;
};

void GrpcClient::ImplDeleter::operator()(Impl* ptr) {
  delete ptr;
}

gxf_result_t GrpcClient::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
    port_, "port", "Target GRPC server port", "Target GRPC server port", 50000U);
  result &= registrar->parameter(
    server_ip_address_, "server_ip_address",
    "Target GRPC server IP", "Target GRPC server IP", std::string("0.0.0.0"));
  result &= registrar->parameter(
    enable_health_check_, "enable_health_check",
    "Enable health check", "Enable gRPC built in heath check stub", false);
  return ToResultCode(result);
}

gxf_result_t GrpcClient::initialize() {
  std::string server_address = this->toIpPort(server_ip_address_.get(), port_.get());
  GXF_LOG_DEBUG("gRPC server endpoint that stubs connect to: %s", server_address.c_str());
  impl_ = std::unique_ptr<Impl, ImplDeleter>(new Impl);
  try {  // create user functional stub
    GXF_LOG_DEBUG("1/3, gRPC API create user functional stub");
    impl_->stub_ = ::gxf::ServiceHub::NewStub(
      CreateChannel(server_address, grpc::InsecureChannelCredentials()));
  } catch (const std::exception& exception) {
    GXF_LOG_ERROR("gRPC failed to create user functional stub, exception: %s",
      exception.what());
    return GXF_IPC_CONNECTION_FAILURE;
  } catch (...) {
    GXF_LOG_ERROR("gRPC failed to create user functional stub");
    return GXF_IPC_CONNECTION_FAILURE;
  }
  if (enable_health_check_.get()) {  // create health check stub
    try {
      GXF_LOG_DEBUG("2/3, gRPC API create built-in health check stub");
      impl_->health_stub_ = grpc::health::v1::Health::NewStub(CreateChannel(server_address,
        grpc::InsecureChannelCredentials()));
    } catch (const std::exception& exception) {
      GXF_LOG_ERROR("gRPC failed to create built-in health check stub, exception: %s",
        exception.what());
      return GXF_IPC_CONNECTION_FAILURE;
    } catch (...) {
      GXF_LOG_ERROR("gRPC failed to create built-in health check stub");
      return GXF_IPC_CONNECTION_FAILURE;
    }
  } else {
    GXF_LOG_DEBUG("2/3, skip create built-in health check stub");
  }

  GXF_LOG_DEBUG("3/3, gRPC API create client stubs returned");
  return GXF_SUCCESS;
}

gxf_result_t GrpcClient::deinitialize() {
  impl_->stub_.reset(nullptr);
  return GXF_SUCCESS;
}

GrpcClient& GrpcClient::changeAddress(const std::string& ip, uint32_t port) {
  impl_->stub_.reset(nullptr);
  std::string server_address = this->toIpPort(ip, port);
  impl_->stub_ = ::gxf::ServiceHub::NewStub(
    CreateChannel(server_address, grpc::InsecureChannelCredentials()));
  GXF_LOG_DEBUG("GrpcClient changed target server address to %s", server_address.c_str());
  return *this;
}

Expected<std::string> GrpcClient::query(
  const std::string& service,
  const std::string& resource
) {
  if (impl_ == nullptr || impl_->stub_ == nullptr) {
    GXF_LOG_ERROR("GrpcClient invalid stub");
    return Unexpected{GXF_ARGUMENT_NULL};
  }
  if (service.empty() || resource.empty()) {
    GXF_LOG_ERROR("Invalid arguments in IPCClient query call");
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  grpc::ClientContext context;
  ::gxf::Request request;
  ::gxf::Response response;
  request.set_service(service);
  request.add_params(resource);
  grpc::Status status = impl_->stub_->SendRequest(&context, request, &response);
  if (status.ok()) {
    const std::string result = response.result();
    GXF_LOG_DEBUG("GrpcClient successfully sent IPC Query request:\n"
    " service: %s\n resource: %s\n result: %s",
      service.c_str(), resource.c_str(), result.c_str());
    return result;
  } else if (status.error_code() == grpc::StatusCode::NOT_FOUND) {
    GXF_LOG_ERROR("GrpcClient went through the server, but service[%s] not found", service.c_str());
    return Unexpected{GXF_IPC_SERVICE_NOT_FOUND};
  } else {
    GXF_LOG_ERROR("GrpcClient failed to get IPC Query response. "
      "service: %s, status: %s, code: %d", service.c_str(),
      status.error_message().c_str(), status.error_code());
    return Unexpected{GXF_IPC_CALL_FAILURE};
  }
}

Expected<void> GrpcClient::action(
  const std::string& service,
  const std::string& resource,
  const std::string& data
) {
  if (impl_ == nullptr || impl_->stub_ == nullptr) {
    GXF_LOG_ERROR("GrpcClient invalid stub");
    return Unexpected{GXF_ARGUMENT_NULL};
  }
  if (service.empty() || resource.empty()) {
    GXF_LOG_ERROR("Invalid arguments in IPCClient query call");
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  grpc::ClientContext context;
  ::gxf::Request request;
  ::gxf::Response response;
  request.set_service(service);
  request.add_params(resource);
  request.add_params(data);
  grpc::Status status = impl_->stub_->SendRequest(&context, request, &response);
  if (status.ok()) {
    // const std::string result = response.result();
    GXF_LOG_DEBUG("GrpcClient successfully sent IPC Action request:\n"
    " service: %s\n resource: %s\n data: %s",
      service.c_str(), resource.c_str(), data.c_str());
    return Success;
  } else {
    GXF_LOG_ERROR("GrpcClient failed to send IPC Action request. "
      "service: %s, status: %s, code: %d", service.c_str(),
      status.error_message().c_str(), status.error_code());
    return Unexpected{GXF_IPC_CALL_FAILURE};
  }
}

static std::string servingStatusToString(
  grpc::health::v1::HealthCheckResponse::ServingStatus status) {
  switch (status) {
    case grpc::health::v1::HealthCheckResponse::SERVING: {
      return "SERVING";
    } break;
    case grpc::health::v1::HealthCheckResponse::NOT_SERVING: {
      return "NOT_SERVING";
    } break;
    case grpc::health::v1::HealthCheckResponse::UNKNOWN: {
      return "UNKNOWN";
    } break;
    default:
      return "INVALID_STATUS";
  }
}

Expected<std::string> GrpcClient::ping(const std::string& service) {
  if (service == kDefaultPingServiceName) {  // IPCServer's own generic ping service
    return this->query(kDefaultPingServiceName, "gxf::IPCServer");
  } else {  // gRPC Health Check Protocol impl
    if (enable_health_check_.get() && impl_->health_stub_) {
      grpc::ClientContext context;
      grpc::health::v1::HealthCheckRequest request;
      request.set_service(service);
      grpc::health::v1::HealthCheckResponse response;
      grpc::Status status = impl_->health_stub_->Check(&context, request, &response);
      if (status.ok()) {
        std::string status_str = servingStatusToString(response.status());
        std::string ping_msg = "service name '" + service + "' health status: " + status_str;
        GXF_LOG_DEBUG("%s", ping_msg.c_str());
        return ping_msg;
      } else if (status.error_message() == "service name unknown" ||
          status.error_code() == grpc::StatusCode::NOT_FOUND) {
        GXF_LOG_ERROR("GRPC Health Check went through the server, but cannot find service[%s]",
          service.c_str());
        return Unexpected{GXF_IPC_SERVICE_NOT_FOUND};
      } else {
        GXF_LOG_ERROR("GrpcClient failed to send grpc::HealthCheckRequest. status: %s",
          status.error_message().c_str());
        return Unexpected{GXF_IPC_CALL_FAILURE};
      }
    } else {
      if (!enable_health_check_.get()) {
        GXF_LOG_ERROR("Calling non-default GRPC Health Check"
          " with param enable_health_check disabled");
      }
      if (!impl_->health_stub_) {
        GXF_LOG_ERROR("Calling non-default GRPC Health Check"
          " with nullptr health check stub");
      }

      return Unexpected{GXF_PARAMETER_NOT_INITIALIZED};
    }
  }
}

}  // namespace gxf
}  // namespace nvidia
