/*
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/ipc/grpc/grpc_server.hpp"
#include "gxf/ipc/grpc/grpc_service.grpc.pb.h"

#include <grpcpp/grpcpp.h>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace nvidia {
namespace gxf {

class ServiceImpl: public ::gxf::ServiceHub::Service {
 public:
  ServiceImpl(GrpcServer& owner): owner_(owner) {}

  grpc::Status SendRequest(grpc::ServerContext* context,
                     const ::gxf::Request* request,
                     ::gxf::Response* response) override {
    auto service = request->service();
    std::vector<std::string> params;
    std::string result;
    for (int i = 0; i < request->params_size(); i++) {
      params.push_back(request->params(i));
    }
    auto maybe = owner_.handleRequest(service, params, result);
    if (!maybe) {
      switch (maybe.error()) {
        case GXF_ARGUMENT_INVALID: {
          return grpc::Status(grpc::INVALID_ARGUMENT, "Invalid argument");
        }
        case GXF_IPC_SERVICE_NOT_FOUND: {
          std::string msg = "GRPC service[" + service + "] not found";
          return grpc::Status(grpc::NOT_FOUND, msg);
        }
        case GXF_FAILURE: {
          return grpc::Status(grpc::INTERNAL, "GXF Failure");
        }
        default: {
          return grpc::Status(grpc::UNKNOWN, "");
        }
      }
    }

    response->set_result(result);

    return grpc::Status::OK;
  }

 private:
  GrpcServer& owner_;
};

class GrpcServer::Impl {
 public:
  std::unique_ptr<grpc::Server> server_;
};
void GrpcServer::ImplDeleter::operator()(Impl* ptr) {
  delete ptr;
}

gxf_result_t GrpcServer::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      port_, "port", "GRPC port for listening", "GRPC port for listening", 50000U);
  result &= registrar->parameter(
      remote_access_, "remote_access",
      "Allow access from remote clients", "Flag for remote access", false);
  result &= registrar->parameter(
      enable_health_check_, "enable_health_check",
      "Enable health check", "Enable gRPC built in heath check service", false);
  return ToResultCode(result);
}

gxf_result_t GrpcServer::initialize() {
  impl_ = std::unique_ptr<Impl, ImplDeleter>(new Impl);

  // Ping handler
  std::string path = "ping";
  IPCServer::QueryHandler handler = std::bind(&GrpcServer::onPing, this, std::placeholders::_1);
  installHandler(query_handlers_, path, handler);

  // Start GRPC server from a separate thread
  std::thread t([this] {
    grpc::EnableDefaultHealthCheckService(enable_health_check_.get());
    ServiceImpl service(*this);
    std::string server_address = remote_access_ ? "0.0.0.0:":"127.0.0.1:";
    server_address += std::to_string(port_);
    grpc::ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    // take the server pointer since we don't want to expose grpc in the header
    impl_->server_ = builder.BuildAndStart();

    // set main entrance service status for gxf IPCServer
    this->setServingStatus(kGrpcMainEntranceService, true);
    // set gxf IPCServer user registered service
    this->setAllRegisteredServiceStatus(true);

    // Wait must be called, otherwise there'll be segmentation fault
    impl_->server_->Wait();
  });
  thread_ = std::move(t);

  return GXF_SUCCESS;
}

gxf_result_t GrpcServer::deinitialize() {
  // set gxf IPCServer user registered service
  this->setAllRegisteredServiceStatus(false);
  // set main entrance service status for gxf IPCServer
  this->setServingStatus(kGrpcMainEntranceService, false);
  impl_->server_->Shutdown();
  thread_.join();
  return GXF_SUCCESS;
}

// registerServer happens before grpc build and start server
Expected<void> GrpcServer::registerService(const IPCServer::Service& service) {
  if (service.type == IPCServer::kQuery) {
    return installHandler(query_handlers_, service.name, service.handler.query);
} else if (service.type == IPCServer::kAction) {
    return installHandler(action_handlers_, service.name, service.handler.action);
} else {
    return Unexpected{GXF_FAILURE};
}
}

Expected<void> GrpcServer::handleRequest(const std::string & service,
                            std::vector<std::string> params,
                            std::string& result) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (query_handlers_.find(service) != query_handlers_.end()) {
    // query request
    if (params.size() != 1) {
      return Unexpected{GXF_ARGUMENT_INVALID};
    }
    auto maybe = query_handlers_[service](params[0]);
    if (!maybe) {
      return Unexpected{GXF_FAILURE};
    }
    result = maybe.value();
  } else if (action_handlers_.find(service) != action_handlers_.end()) {
    // action request
    if (params.size() != 2) {
      return Unexpected{GXF_ARGUMENT_INVALID};
    }
    auto& resource = params[0];
    auto& data = params[1];
    auto maybe = action_handlers_[service](resource, data);
    if (!maybe) {
      return Unexpected{GXF_FAILURE};
    }
  } else {
    // set service status to NOT_SERVING, as server cannot find the service anymore
    this->setServingStatus(service, false);
    return Unexpected{GXF_IPC_SERVICE_NOT_FOUND};
  }

  return Success;
}

Expected<void> GrpcServer::setServingStatus(const std::string& service_name, bool serving) {
  if (!enable_health_check_.get()) {
    GXF_LOG_DEBUG("skip setting service status, as health check is disabled");
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  if (impl_->server_ == nullptr) {
    GXF_LOG_ERROR("grpc::Server pointer is nullptr");
    return Unexpected{GXF_NULL_POINTER};
  }
  grpc::HealthCheckServiceInterface* health_service =
    impl_->server_->GetHealthCheckService();
  if (!health_service) {
    GXF_LOG_ERROR("nullptr grpc::HealthCheckServiceInterface");
    return Unexpected{GXF_NULL_POINTER};
  }
  // true -> grpc::HealthCheckResponse::SERVING
  // false -> grpc::HealthCheckResponse::NOT_SERVING
  if (service_name.empty()) {
    GXF_LOG_DEBUG("GrpcServer setting all services serving to [%d]", serving);
    health_service->SetServingStatus(serving);
  } else {
    GXF_LOG_DEBUG("GrpcServer setting service[name: %s] serving to [%d]",
      service_name.c_str(), serving);
    health_service->SetServingStatus(service_name, serving);
  }

  return Success;
}

Expected<void> GrpcServer::setAllRegisteredServiceStatus(bool serving) {
  for (const auto& pair : query_handlers_) {
    if (pair.second) {
      this->setServingStatus(pair.first, serving);
    }
  }
  for (const auto& pair : action_handlers_) {
    if (pair.second) {
      this->setServingStatus(pair.first, serving);
    }
  }

  return Success;
}

}  // namespace gxf
}  // namespace nvidia
