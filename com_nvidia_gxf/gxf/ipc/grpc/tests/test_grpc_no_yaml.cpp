/*
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.

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
}  // namespace

class TestGrpc : public ::testing::Test {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfSetSeverity(&context, GXF_SEVERITY_DEBUG));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
    GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info_sch, &eid_sch));
    GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info_svr, &eid_server));
    GXF_ASSERT_SUCCESS(GxfCreateEntity(context, &entity_create_info_clt, &eid_client));
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::GreedyScheduler", &tid_sch));
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::RealtimeClock", &tid_clock));
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::GrpcServer", &tid_server));
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::GrpcClient", &tid_client));

    GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid_sch, tid_sch, "scheduler", &cid_sch));
    GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid_sch, tid_clock, "clock", &cid_clock));
    GXF_ASSERT_SUCCESS(GxfParameterSetHandle(context, cid_sch, "clock", cid_clock));
    GXF_ASSERT_SUCCESS(GxfParameterSetInt64(context, cid_sch, "max_duration_ms", 1000000));

    GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid_server, tid_server, "grpc_server", &cid_server));
    GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid_client, tid_client, "grpc_client", &cid_client));
    GXF_ASSERT_SUCCESS(GxfParameterSetInt32(context, cid_server, "port", 50001));
    GXF_ASSERT_SUCCESS(GxfParameterSetInt32(context, cid_client, "port", 50001));
  }

  void TearDown() {
    // GXF_ASSERT_SUCCESS(GxfEntityDestroy(context, eid_sch));
    // GXF_ASSERT_SUCCESS(GxfEntityDestroy(context, eid_server));
    // GXF_ASSERT_SUCCESS(GxfEntityDestroy(context, eid_client));
    GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
  }

 protected:
  gxf_context_t context = kNullContext;
  const GxfLoadExtensionsInfo info{kExtensions, 3, nullptr, 0, nullptr};
  // const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  const GxfEntityCreateInfo entity_create_info_sch = {"scheduler", GXF_ENTITY_CREATE_PROGRAM_BIT};
  const GxfEntityCreateInfo entity_create_info_svr = {"server", GXF_ENTITY_CREATE_PROGRAM_BIT};
  const GxfEntityCreateInfo entity_create_info_clt = {"client", GXF_ENTITY_CREATE_PROGRAM_BIT};
  gxf_uid_t eid_sch, eid_server, eid_client = kNullUid;
  gxf_tid_t tid_sch, tid_clock, tid_server, tid_client = GxfTidNull();
  gxf_uid_t cid_sch, cid_clock, cid_server, cid_client = kNullUid;
};

TEST_F(TestGrpc, ClientToServerNoYaml) {
  ASSERT_EQ(GxfGraphActivate(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphRunAsync(context), GXF_SUCCESS);
  GrpcPingClient client(grpc::CreateChannel(
      "localhost:50000", grpc::InsecureChannelCredentials()));
  for (auto i = 0; i < 100; i++) {
    ASSERT_EQ(client.Ping(), true);
  }

  ASSERT_EQ(GxfGraphWait(context), GXF_SUCCESS);
}

}  // namespace gxf
}  // namespace nvidia
