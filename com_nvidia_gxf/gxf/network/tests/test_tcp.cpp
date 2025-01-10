/*
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gtest/gtest.h"

#include <chrono>
#include <cstring>
#include <thread>
#include <utility>

#include "gxf/network/tcp_client_socket.hpp"
#include "gxf/network/tcp_server_socket.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/double_buffer_receiver.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/test/components/mock_receiver.hpp"
#include "gxf/test/components/tensor_generator.hpp"

namespace nvidia {
namespace gxf {

namespace {

constexpr const char* kAddress = "127.0.0.1";
constexpr uint16_t kPort = 7000;
constexpr size_t kBlockSize = 1024;

constexpr const char* kExtensions[] = {
    "gxf/std/libgxf_std.so",
    "gxf/serialization/libgxf_serialization.so",
    "gxf/network/libgxf_network.so",
    "gxf/sample/libgxf_sample.so",
    "gxf/test/extensions/libgxf_test.so",
};

constexpr GxfLoadExtensionsInfo kExtensionInfo{kExtensions, 5, nullptr, 0, nullptr};

}  // namespace

TEST(TestTcp, TcpClientSocket) {
  byte block[kBlockSize];
  std::memset(block, 0xAA, kBlockSize);
  byte buffer[kBlockSize];
  std::memset(buffer, 0x00, kBlockSize);

  TcpClientSocket client_socket;
  ASSERT_TRUE(client_socket.open());
  ASSERT_FALSE(client_socket.connected());

  ASSERT_FALSE(client_socket.connect(kAddress, kPort));
  ASSERT_FALSE(client_socket.connected());

  ASSERT_FALSE(client_socket.write(block, kBlockSize));
  ASSERT_FALSE(client_socket.read(buffer, kBlockSize));

  ASSERT_TRUE(client_socket.close());
}

TEST(TestTcp, TcpServerSocket) {
  uint8_t block1[kBlockSize];
  std::memset(block1, 0xAA, kBlockSize);
  uint8_t block2[kBlockSize];
  std::memset(block2, 0xBB, kBlockSize);
  uint8_t block3[kBlockSize];
  std::memset(block3, 0xCC, kBlockSize);
  uint8_t block4[kBlockSize];
  std::memset(block4, 0xDD, kBlockSize);
  uint8_t buffer[kBlockSize];
  std::memset(buffer, 0x00, kBlockSize);

  TcpServerSocket server_socket = TcpServerSocket(kAddress, kPort);
  ASSERT_TRUE(server_socket.open());

  TcpClientSocket client1_socket;
  ASSERT_TRUE(client1_socket.open());

  ASSERT_TRUE(client1_socket.connect(kAddress, kPort));
  auto result = server_socket.connect();
  ASSERT_TRUE(result);
  TcpClientSocket client2_socket = std::move(result.value());

  ASSERT_TRUE(client1_socket.connected());
  ASSERT_TRUE(client2_socket.connected());

  ASSERT_FALSE(client2_socket.available());
  ASSERT_TRUE(client1_socket.write(block1, kBlockSize));
  ASSERT_TRUE(client1_socket.write(block2, kBlockSize));
  ASSERT_TRUE(client2_socket.available());
  ASSERT_TRUE(client2_socket.read(buffer, kBlockSize));
  ASSERT_EQ(std::memcmp(buffer, block1, kBlockSize), 0);
  ASSERT_TRUE(client2_socket.read(buffer, kBlockSize));
  ASSERT_EQ(std::memcmp(buffer, block2, kBlockSize), 0);

  ASSERT_FALSE(client1_socket.available());
  ASSERT_TRUE(client2_socket.write(block3, kBlockSize));
  ASSERT_TRUE(client2_socket.write(block4, kBlockSize));
  ASSERT_TRUE(client1_socket.available());
  ASSERT_TRUE(client1_socket.read(buffer, kBlockSize));
  ASSERT_EQ(std::memcmp(buffer, block3, kBlockSize), 0);
  ASSERT_TRUE(client1_socket.read(buffer, kBlockSize));
  ASSERT_EQ(std::memcmp(buffer, block4, kBlockSize), 0);

  ASSERT_TRUE(client1_socket.close());
  ASSERT_FALSE(client1_socket.connected());
  ASSERT_TRUE(client2_socket.connected());

  ASSERT_FALSE(client2_socket.read(buffer, kBlockSize));
  ASSERT_FALSE(client1_socket.connected());
  ASSERT_FALSE(client2_socket.connected());

  ASSERT_TRUE(client2_socket.close());
  ASSERT_TRUE(server_socket.close());
}

TEST(TestTcp, ClientToServer) {
  gxf_context_t context1;
  ASSERT_EQ(GxfContextCreate(&context1), GXF_SUCCESS);

  gxf_context_t context2;
  ASSERT_EQ(GxfContextCreate(&context2), GXF_SUCCESS);

  ASSERT_EQ(GxfLoadExtensions(context1, &kExtensionInfo), GXF_SUCCESS);
  ASSERT_EQ(GxfLoadExtensions(context2, &kExtensionInfo), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphLoadFile(context1, "gxf/network/tests/test_tcp_client_source.yaml"),
            GXF_SUCCESS);
  ASSERT_EQ(GxfGraphLoadFile(context2, "gxf/network/tests/test_tcp_server_sink.yaml"), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphActivate(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphActivate(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphRunAsync(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphRunAsync(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphWait(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphWait(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphDeactivate(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphDeactivate(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfContextDestroy(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context2), GXF_SUCCESS);
}

TEST(TestTcp, ServerToClient) {
  gxf_context_t context1;
  ASSERT_EQ(GxfContextCreate(&context1), GXF_SUCCESS);

  gxf_context_t context2;
  ASSERT_EQ(GxfContextCreate(&context2), GXF_SUCCESS);

  ASSERT_EQ(GxfLoadExtensions(context1, &kExtensionInfo), GXF_SUCCESS);
  ASSERT_EQ(GxfLoadExtensions(context2, &kExtensionInfo), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphLoadFile(context1, "gxf/network/tests/test_tcp_server_source.yaml"),
            GXF_SUCCESS);
  ASSERT_EQ(GxfGraphLoadFile(context2, "gxf/network/tests/test_tcp_client_sink.yaml"), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphActivate(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphActivate(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphRunAsync(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphRunAsync(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphWait(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphWait(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphDeactivate(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphDeactivate(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfContextDestroy(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context2), GXF_SUCCESS);
}

TEST(TestTcp, ServerToClientAsync) {
  gxf_context_t context1;
  gxf_context_t shared_context;

  ASSERT_EQ(GxfContextCreate(&context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGetSharedContext(context1, &shared_context), GXF_SUCCESS);

  gxf_context_t context2;
  ASSERT_EQ(GxfContextCreateShared(shared_context, &context2), GXF_SUCCESS);

  ASSERT_EQ(GxfLoadExtensions(context1, &kExtensionInfo), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphLoadFile(context1, "gxf/network/tests/test_tcp_server_source_async.yaml"),
            GXF_SUCCESS);

  ASSERT_EQ(GxfGraphLoadFileExtended(context2, "gxf/network/tests/test_tcp_client_sink_async.yaml",
                                     "shared_ctx"),
            GXF_SUCCESS);

  ASSERT_EQ(GxfGraphActivate(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphActivate(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphRunAsync(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphRunAsync(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphWait(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphWait(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphDeactivate(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphDeactivate(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfContextDestroy(context2), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context1), GXF_SUCCESS);
}

TEST(TestTcp, Bidirectional) {
  gxf_context_t context1;
  ASSERT_EQ(GxfContextCreate(&context1), GXF_SUCCESS);

  gxf_context_t context2;
  ASSERT_EQ(GxfContextCreate(&context2), GXF_SUCCESS);

  ASSERT_EQ(GxfLoadExtensions(context1, &kExtensionInfo), GXF_SUCCESS);
  ASSERT_EQ(GxfLoadExtensions(context2, &kExtensionInfo), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphLoadFile(context1, "gxf/network/tests/test_tcp_client_bidirectional.yaml"),
            GXF_SUCCESS);
  ASSERT_EQ(GxfGraphLoadFile(context2, "gxf/network/tests/test_tcp_server_bidirectional.yaml"),
            GXF_SUCCESS);

  ASSERT_EQ(GxfGraphActivate(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphActivate(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphRunAsync(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphRunAsync(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphWait(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphWait(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphDeactivate(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphDeactivate(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfContextDestroy(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context2), GXF_SUCCESS);
}

TEST(TestTcp, ClientReconnection) {
  gxf_context_t context1;
  gxf_context_t context2;

  ASSERT_EQ(GxfContextCreate(&context1), GXF_SUCCESS);
  ASSERT_EQ(GxfContextCreate(&context2), GXF_SUCCESS);

  ASSERT_EQ(GxfLoadExtensions(context1, &kExtensionInfo), GXF_SUCCESS);
  ASSERT_EQ(GxfLoadExtensions(context2, &kExtensionInfo), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphLoadFile(context1, "gxf/network/tests/test_tcp_client_source.yaml"),
            GXF_SUCCESS);
  ASSERT_EQ(
      GxfGraphLoadFile(context2, "gxf/network/tests/test_tcp_client_reconnection_server.yaml"),
      GXF_SUCCESS);

  ASSERT_EQ(GxfGraphActivate(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphActivate(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphRunAsync(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphRunAsync(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphWait(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphDeactivate(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context1), GXF_SUCCESS);

  gxf_context_t context3;
  ASSERT_EQ(GxfContextCreate(&context3), GXF_SUCCESS);
  ASSERT_EQ(GxfLoadExtensions(context3, &kExtensionInfo), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphLoadFile(context3, "gxf/network/tests/test_tcp_client_source.yaml"),
            GXF_SUCCESS);

  ASSERT_EQ(GxfGraphActivate(context3), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphRunAsync(context3), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphWait(context3), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphWait(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphDeactivate(context3), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphDeactivate(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfContextDestroy(context3), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context2), GXF_SUCCESS);
}

TEST(TestTcp, ServerReconnection) {
  gxf_context_t context1;
  gxf_context_t context2;

  ASSERT_EQ(GxfContextCreate(&context1), GXF_SUCCESS);
  ASSERT_EQ(GxfContextCreate(&context2), GXF_SUCCESS);

  ASSERT_EQ(GxfLoadExtensions(context1, &kExtensionInfo), GXF_SUCCESS);
  ASSERT_EQ(GxfLoadExtensions(context2, &kExtensionInfo), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphLoadFile(context1, "gxf/network/tests/test_tcp_server_source.yaml"),
            GXF_SUCCESS);
  ASSERT_EQ(
      GxfGraphLoadFile(context2, "gxf/network/tests/test_tcp_server_reconnection_client.yaml"),
      GXF_SUCCESS);

  ASSERT_EQ(GxfGraphActivate(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphActivate(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphRunAsync(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphRunAsync(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphWait(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphDeactivate(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context1), GXF_SUCCESS);

  gxf_context_t context3;
  ASSERT_EQ(GxfContextCreate(&context3), GXF_SUCCESS);
  ASSERT_EQ(GxfLoadExtensions(context3, &kExtensionInfo), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphLoadFile(context3, "gxf/network/tests/test_tcp_server_source.yaml"),
            GXF_SUCCESS);

  ASSERT_EQ(GxfGraphActivate(context3), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphRunAsync(context3), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphWait(context3), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphWait(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphDeactivate(context3), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphDeactivate(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfContextDestroy(context3), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context2), GXF_SUCCESS);
}

TEST(TestTcp, ServerReceiverPush) {
  gxf_context_t context1;
  gxf_context_t context2;

  ASSERT_EQ(GxfContextCreate(&context1), GXF_SUCCESS);
  ASSERT_EQ(GxfContextCreate(&context2), GXF_SUCCESS);

  ASSERT_EQ(GxfLoadExtensions(context1, &kExtensionInfo), GXF_SUCCESS);
  ASSERT_EQ(GxfLoadExtensions(context2, &kExtensionInfo), GXF_SUCCESS);

  ASSERT_EQ(
      GxfGraphLoadFile(context1, "gxf/network/tests/test_tcp_server_receiver_push_server.yaml"),
      GXF_SUCCESS);
  ASSERT_EQ(
      GxfGraphLoadFile(context2, "gxf/network/tests/test_tcp_server_receiver_push_client.yaml"),
      GXF_SUCCESS);

  ASSERT_EQ(GxfGraphActivate(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphActivate(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphRunAsync(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphRunAsync(context2), GXF_SUCCESS);

  // Find tcp_server eid
  gxf_uid_t server_eid;
  GXF_ASSERT_SUCCESS(GxfEntityFind(context1, "server", &server_eid));
  // Get handle to tcp_server/allocator
  gxf_tid_t allocator_tid;
  GXF_ASSERT_SUCCESS(
      GxfComponentTypeId(context1, nvidia::TypenameAsString<Allocator>(), &allocator_tid));
  gxf_uid_t allocator_cid;
  GXF_ASSERT_SUCCESS(
      GxfComponentFind(context1, server_eid, allocator_tid, "allocator", nullptr, &allocator_cid));
  auto allocator = Handle<Allocator>::Create(context1, allocator_cid);
  ASSERT_TRUE(allocator.has_value());
  // Get handle to tcp_server/channel
  gxf_tid_t receiver_tid;
  GXF_ASSERT_SUCCESS(GxfComponentTypeId(context1, nvidia::TypenameAsString<DoubleBufferReceiver>(),
                                        &receiver_tid));
  gxf_uid_t receiver_cid;
  GXF_ASSERT_SUCCESS(
      GxfComponentFind(context1, server_eid, receiver_tid, "channel", nullptr, &receiver_cid));
  auto receiver = Handle<DoubleBufferReceiver>::Create(context1, receiver_cid);
  ASSERT_TRUE(receiver.has_value());

  std::vector<int32_t> shape = {2, 2, 2};
  uint32_t rank = shape.size();
  auto min = std::numeric_limits<test::TensorGenerator::DataType>::min();
  auto max = std::numeric_limits<test::TensorGenerator::DataType>::max();
  std::uniform_real_distribution<test::TensorGenerator::DataType> distribution(min, max);
  std::default_random_engine generator;

  for (int i = 0; i < 1000; i++) {
    auto message = Entity::New(context1);
    ASSERT_TRUE(message.has_value());

    auto tensor = message.value().add<Tensor>("tensor");
    ASSERT_TRUE(tensor.has_value());

    std::array<int32_t, Shape::kMaxRank> dims;
    std::copy(std::begin(shape), std::end(shape), std::begin(dims));
    auto result = tensor.value()->reshape<test::TensorGenerator::DataType>(
        Shape(dims, rank), MemoryStorageType(0), allocator.value());
    ASSERT_TRUE(result.has_value());

    std::vector<test::TensorGenerator::DataType> elements;
    for (size_t idx = 0; idx < tensor.value()->element_count(); idx++) {
      elements.push_back(distribution(generator));
    }

    const cudaMemcpyKind operation = (tensor.value()->storage_type() == MemoryStorageType::kHost ||
                                      tensor.value()->storage_type() == MemoryStorageType::kSystem)
                                         ? cudaMemcpyHostToHost
                                         : cudaMemcpyHostToDevice;
    const cudaError_t error =
        cudaMemcpy(tensor.value()->pointer(), elements.data(), tensor.value()->size(), operation);
    ASSERT_EQ(error, cudaSuccess);

    ASSERT_EQ(receiver.value()->push_abi(message.value().eid()), GXF_SUCCESS);
  }

  ASSERT_EQ(GxfGraphWait(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphWait(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphDeactivate(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphDeactivate(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfContextDestroy(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context2), GXF_SUCCESS);
}

TEST(TestTcp, ServerUnconnectedInterrupt) {
  using namespace std::chrono_literals;
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  ASSERT_EQ(GxfLoadExtensions(context, &kExtensionInfo), GXF_SUCCESS);

  ASSERT_EQ(
      GxfGraphLoadFile(context, "gxf/network/tests/test_tcp_server_unconnected_interrupt.yaml"),
      GXF_SUCCESS);

  ASSERT_EQ(GxfGraphActivate(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphRunAsync(context), GXF_SUCCESS);
  std::this_thread::sleep_for(1s);
  ASSERT_EQ(GxfGraphInterrupt(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphWait(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphDeactivate(context), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(TestTcp, ClientUnconnectedInterrupt) {
  using namespace std::chrono_literals;
  gxf_context_t context;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  ASSERT_EQ(GxfLoadExtensions(context, &kExtensionInfo), GXF_SUCCESS);

  ASSERT_EQ(
      GxfGraphLoadFile(context, "gxf/network/tests/test_tcp_client_unconnected_interrupt.yaml"),
      GXF_SUCCESS);

  ASSERT_EQ(GxfGraphActivate(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphRunAsync(context), GXF_SUCCESS);
  std::this_thread::sleep_for(1s);
  ASSERT_EQ(GxfGraphInterrupt(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphWait(context), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphDeactivate(context), GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(TestTcp, ClientServerConnectedInterrupt) {
  using namespace std::chrono_literals;

  gxf_context_t context1;
  ASSERT_EQ(GxfContextCreate(&context1), GXF_SUCCESS);

  gxf_context_t context2;
  ASSERT_EQ(GxfContextCreate(&context2), GXF_SUCCESS);

  ASSERT_EQ(GxfLoadExtensions(context1, &kExtensionInfo), GXF_SUCCESS);
  ASSERT_EQ(GxfLoadExtensions(context2, &kExtensionInfo), GXF_SUCCESS);

  ASSERT_EQ(
      GxfGraphLoadFile(context1, "gxf/network/tests/test_tcp_client_connected_interrupt.yaml"),
      GXF_SUCCESS);
  ASSERT_EQ(
      GxfGraphLoadFile(context2, "gxf/network/tests/test_tcp_server_connected_interrupt.yaml"),
      GXF_SUCCESS);

  ASSERT_EQ(GxfGraphActivate(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphActivate(context2), GXF_SUCCESS);

  ASSERT_EQ(GxfGraphRunAsync(context1), GXF_SUCCESS);
  ASSERT_EQ(GxfGraphRunAsync(context2), GXF_SUCCESS);
  std::this_thread::sleep_for(200ms);

  // Interrupt client graph while it's attempting to deserialize messages from server.
  GxfGraphInterrupt(context1);
  GxfGraphWait(context1);
  GxfGraphDeactivate(context1);
  GxfContextDestroy(context1);

  GxfGraphWait(context2);
  GxfGraphDeactivate(context2);
  GxfContextDestroy(context2);
}

}  // namespace gxf
}  // namespace nvidia
