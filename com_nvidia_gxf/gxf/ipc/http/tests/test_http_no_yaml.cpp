/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"
#include "gxf/ipc/http/http_ipc_client.hpp"
#include "gxf/ipc/http/http_server.hpp"

namespace nvidia {
namespace gxf {

namespace {
  constexpr const char* kExtensions[] = {
    "gxf/std/libgxf_std.so",
    "gxf/ipc/http/libgxf_http.so",
    "gxf/test/extensions/libgxf_test.so",
  };
  constexpr GxfLoadExtensionsInfo kExtensionInfo{kExtensions, 3, nullptr, 0, nullptr};
}  // namespace

class TestHttp : public ::testing::Test {
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
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::HttpServer", &tid_server));
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::HttpIPCClient", &tid_client));

    GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid_sch, tid_sch, "scheduler", &cid_sch));
    GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid_sch, tid_clock, "clock", &cid_clock));
    GXF_ASSERT_SUCCESS(GxfParameterSetHandle(context, cid_sch, "clock", cid_clock));
    GXF_ASSERT_SUCCESS(GxfParameterSetInt64(context, cid_sch, "max_duration_ms", 1000000));

    GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid_server, tid_server, "http_server", &cid_server));
    GXF_ASSERT_SUCCESS(GxfComponentAdd(context, eid_client, tid_client, "http_client", &cid_client));
    GXF_ASSERT_SUCCESS(GxfParameterSetUInt32(context, cid_server, "port", 50001U));
    GXF_ASSERT_SUCCESS(GxfParameterSetUInt32(context, cid_client, "port", 50001U));
  }

  void TearDown() {
    GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
  }

  Expected<std::string> onMockQuery(const std::string& resource) {
    GXF_LOG_INFO("Service querying on resource: %s", resource.c_str());
    return "{\"key1\": \"value1_mock\", \"key2\": \"value2_mock\"}";
  }
  Expected<void> onMockAction(const std::string& resource,
                                              const std::string& data) {
    Expected<void> result;
    GXF_LOG_INFO("Service taking action on resource: %s, with data: %s\n", resource.c_str(),
                data.c_str());
    return result;
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

TEST_F(TestHttp, ClientToServerPing) {
  ASSERT_EQ(GxfGraphActivate(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphRunAsync(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphWait(context), GXF_SUCCESS);

  auto maybe_client = nvidia::gxf::Handle<nvidia::gxf::HttpIPCClient>::Create(context, cid_client);
  ASSERT_TRUE(maybe_client.has_value());
  auto client = maybe_client.value();
  Expected<std::string> response = client->ping();
  bool success;
  if (!response) {
    success = false;
  } else {
    GXF_LOG_INFO("client to server ping response: %s", response.value().c_str());
    success = true;
  }
  ASSERT_TRUE(success);
}

TEST_F(TestHttp, ClientToServerQuery) {
  IPCServer::Service service_sample_query = {
    "test_query_service",
    IPCServer::kQuery,
    {.query = std::bind(&TestHttp::onMockQuery, this, std::placeholders::_1)}
  };
  auto maybe_server = nvidia::gxf::Handle<nvidia::gxf::HttpServer>::Create(context, cid_server);
  ASSERT_TRUE(maybe_server.has_value());
  auto server = maybe_server.value();
  server->registerService(service_sample_query);

  ASSERT_EQ(GxfGraphActivate(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphRunAsync(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphWait(context), GXF_SUCCESS);

  auto maybe_client = nvidia::gxf::Handle<nvidia::gxf::HttpIPCClient>::Create(context, cid_client);
  ASSERT_TRUE(maybe_client.has_value());
  auto client = maybe_client.value();
  Expected<std::string> response = client->query("test_query_service", "menu");
  bool success;
  if (!response) {
    success = false;
  } else {
    GXF_LOG_INFO("client to server query response: %s", response.value().c_str());
    success = true;
  }
  ASSERT_TRUE(success);
}

TEST_F(TestHttp, ClientToServerAction) {
  IPCServer::Service service_sample_action = {
    "test_action_service",
    IPCServer::kAction,
    {.action = std::bind(&TestHttp::onMockAction, this, std::placeholders::_1,
                          std::placeholders::_2)}
  };
  auto maybe_server = nvidia::gxf::Handle<nvidia::gxf::HttpServer>::Create(context, cid_server);
  ASSERT_TRUE(maybe_server.has_value());
  auto server = maybe_server.value();
  server->registerService(service_sample_action);

  ASSERT_EQ(GxfGraphActivate(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphRunAsync(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphWait(context), GXF_SUCCESS);

  auto maybe_client = nvidia::gxf::Handle<nvidia::gxf::HttpIPCClient>::Create(context, cid_client);
  ASSERT_TRUE(maybe_client.has_value());
  auto client = maybe_client.value();
  Expected<void> response = client->action(
    "test_action_service", "menu", "{\"key1\": \"value1\", \"key2\": \"value2\"}");
  bool success;
  if (!response) {
    success = false;
  } else {
    GXF_LOG_INFO("client to server action success");
    success = true;
  }
  ASSERT_TRUE(success);
}

TEST_F(TestHttp, ClientToServerChangeAddress) {
  uint32_t server_port = 50002U;
  GXF_ASSERT_SUCCESS(GxfParameterSetUInt32(context, cid_server, "port", server_port));
  ASSERT_EQ(GxfGraphActivate(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphRunAsync(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphWait(context), GXF_SUCCESS);

  auto maybe_client = nvidia::gxf::Handle<nvidia::gxf::HttpIPCClient>::Create(context, cid_client);
  ASSERT_TRUE(maybe_client.has_value());
  auto client = maybe_client.value();
  Expected<std::string> response = client->changeAddress("localhost", server_port).ping();
  bool success;
  if (!response) {
    success = false;
  } else {
    GXF_LOG_INFO("client to server at port[%d] ping response: %s",
      server_port, response.value().c_str());
    success = true;
  }
  ASSERT_TRUE(success);
}

}  // namespace gxf
}  // namespace nvidia
