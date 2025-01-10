/*
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gtest/gtest.h"
#include "gxf/core/gxf.h"
#include <grpcpp/grpcpp.h>
#include "gxf/ipc/grpc/grpc_client.hpp"
#include "gxf/ipc/grpc/grpc_service.grpc.pb.h"
#include "gxf/ipc/grpc/grpc_server.hpp"

namespace nvidia {
namespace gxf {

class GrpcPingClient {
 public:
   explicit GrpcPingClient(std::shared_ptr<grpc::ChannelInterface> channel)
       : stub_(::gxf::ServiceHub::NewStub(channel)) {}

   bool Ping() {
     ::gxf::Request ping;
     ::gxf::Response response;
     ping.set_service("ping");
     ping.add_params("hello");

     grpc::ClientContext context;
     grpc::Status status = stub_->SendRequest(&context, ping, &response);
     if (!status.ok()) {
        std::cout << status.error_message() << std::endl;
        return false;
     }

     std::cout << response.result() << std::endl;
     return true;
   }
 private:
   std::unique_ptr<::gxf::ServiceHub::Stub> stub_;
};

namespace {
  constexpr const char* kExtensions[] = {
    "gxf/std/libgxf_std.so",
    "gxf/ipc/grpc/libgxf_grpc.so",
    "gxf/test/extensions/libgxf_test.so",
  };
  constexpr GxfLoadExtensionsInfo kExtensionInfo{kExtensions, 3, nullptr, 0, nullptr};
}

TEST(TestGrpc, ClientToServer) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfSetSeverity(&context, GXF_SEVERITY_DEBUG));
  ASSERT_EQ(GxfLoadExtensions(context, &kExtensionInfo), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphLoadFile(context, "gxf/ipc/grpc/tests/test_grpc_server.yaml"),
            GXF_SUCCESS);
  ASSERT_EQ(GxfGraphActivate(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphRunAsync(context), GXF_SUCCESS);

  GrpcPingClient client(grpc::CreateChannel(
      "localhost:50000", grpc::InsecureChannelCredentials()));
  for (auto i = 0; i < 100; i++) {
    ASSERT_EQ(client.Ping(), true);
  }

  ASSERT_EQ(GxfGraphWait(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphDeactivate(context), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(TestGrpc, Client) {
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
  ASSERT_EQ(GxfLoadExtensions(context, &kExtensionInfo), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphLoadFile(context, "gxf/ipc/grpc/tests/test_grpc_server.yaml"),
            GXF_SUCCESS);
  ASSERT_EQ(GxfGraphActivate(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphRunAsync(context), GXF_SUCCESS);

  gxf_uid_t eid, cid;
  gxf_tid_t tid;
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "client_entity", &eid));
  GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::GrpcClient", &tid));
  GXF_ASSERT_SUCCESS(GxfComponentFind(context, eid, tid, "grpc_client", nullptr, &cid));
  auto maybe_client = nvidia::gxf::Handle<nvidia::gxf::GrpcClient>::Create(context, cid);
  ASSERT_TRUE(maybe_client.has_value());
  auto client = maybe_client.value();
  ASSERT_EQ(client->changeAddress("localhost", 50000).query("ping", "server"), "server is good\n");

  ASSERT_EQ(GxfGraphWait(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphDeactivate(context), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

}
}