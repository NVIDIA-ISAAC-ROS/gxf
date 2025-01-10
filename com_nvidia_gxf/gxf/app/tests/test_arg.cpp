/*
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <chrono>
#include <climits>
#include <cstdint>
#include <cstring>

#include <iostream>
#include <thread>

#include "common/assert.hpp"
#include "gtest/gtest.h"

#include "gxf/app/arg.hpp"
#include "gxf/app/arg_parse.hpp"
#include "gxf/core/filepath.hpp"
#include "gxf/sample/ping_rx.hpp"
#include "gxf/sample/ping_tx.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/block_memory_pool.hpp"
#include "gxf/std/double_buffer_receiver.hpp"
#include "gxf/std/double_buffer_transmitter.hpp"
#include "gxf/std/forward.hpp"
#include "gxf/std/unbounded_allocator.hpp"

namespace {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";
const static gxf_tid_t TransmitterTID {0xc30cc60f0db2409d, 0x92b6b2db92e02cce};
const static gxf_tid_t ReceiverTID {0xa47d2f62245f40fc, 0x90b75dc78ff2437e};
const static gxf_tid_t DoubleBufferTxTID {0x0c3c0ec777f14389, 0xaef16bae85bddc13};
const static gxf_tid_t DoubleBufferRxTID {0xa47d2f62245f40fc, 0x90b75dc78ff2437e};

}  // namespace

class GxfArg : public ::testing::Test {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GxfSetSeverity(context, GXF_SEVERITY_INFO);
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  }

  void TearDown() { GXF_ASSERT_SUCCESS(GxfContextDestroy(context)); }

 protected:
  gxf_context_t context = kNullContext;
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  const GxfEntityCreateInfo entity_create_info = {0};
};

namespace nvidia {
namespace gxf {

TEST_F(GxfArg, Basic) {
  nvidia::gxf::Arg empty("empty");
  ASSERT_FALSE(empty.has_value());

  nvidia::gxf::Arg simple_int = Arg("int", int64_t{10});
  ASSERT_EQ(std::any_cast<int64_t>(simple_int.value()), 10);
  ASSERT_EQ(simple_int.as<int64_t>(), 10);
  ASSERT_EQ(simple_int.arg_type_name(), "Int64");
  // ASSERT_EQ(simple_int.yaml_node(), YAML::Node(10));

  nvidia::gxf::Arg simple_string = Arg("string", std::string("simple string"));
  ASSERT_EQ(std::any_cast<std::string>(simple_string.value()), "simple string");
  ASSERT_EQ(simple_string.as<std::string>(), "simple string");
  ASSERT_EQ(simple_string.arg_type_name(), "String");
  // ASSERT_EQ(simple_string.yaml_node(), YAML::Node("simple string"));

  nvidia::gxf::Arg simple_fp = Arg("filepath", FilePath("simple file path"));
  ASSERT_EQ(std::any_cast<FilePath>(simple_fp.value()), "simple file path");
  ASSERT_EQ(simple_fp.as<FilePath>(), "simple file path");
  ASSERT_EQ(simple_fp.arg_type_name(), "File");

  nvidia::gxf::Arg simple_vector = Arg("vector", std::vector<int>({1, 2, 3}));
  GXF_ASSERT_EQ(std::any_cast<std::vector<int>>(simple_vector.value())[0], 1);
  GXF_ASSERT_EQ(std::any_cast<std::vector<int>>(simple_vector.value())[1], 2);
  GXF_ASSERT_EQ(std::any_cast<std::vector<int>>(simple_vector.value())[2], 3);
  ASSERT_EQ(simple_vector.arg_type_name(), "Int32");
  // ASSERT_EQ(simple_vector.yaml_node(), YAML::Node(std::vector<int>({1, 2, 3})));
}

TEST_F(GxfArg, ArgInfo) {
  ArgInfo pod_info;
  ArgOverride<float>::apply(pod_info);
  ASSERT_EQ(pod_info.type, GXF_PARAMETER_TYPE_FLOAT32);
  ASSERT_EQ(pod_info.type_name, "Float32");
  ASSERT_EQ(pod_info.rank, 0);
  ASSERT_EQ(pod_info.shape[0], 0);

  ArgInfo string_info;
  ArgOverride<std::string>::apply(string_info);
  ASSERT_EQ(string_info.type, GXF_PARAMETER_TYPE_STRING);
  ASSERT_EQ(string_info.type_name, "String");
  ASSERT_EQ(string_info.rank, 0);
  ASSERT_EQ(string_info.shape[0], 0);

  ArgInfo filepath_info;
  ArgOverride<FilePath>::apply(filepath_info);
  ASSERT_EQ(filepath_info.type, GXF_PARAMETER_TYPE_FILE);
  ASSERT_EQ(filepath_info.type_name, "File");
  ASSERT_EQ(filepath_info.rank, 0);
  ASSERT_EQ(filepath_info.shape[0], 0);

  ArgInfo handle_info;
  ArgOverride<Handle<Allocator>>::apply(handle_info);
  ASSERT_EQ(handle_info.type, GXF_PARAMETER_TYPE_HANDLE);
  ASSERT_EQ(handle_info.type_name, "nvidia::gxf::Allocator");
  ASSERT_EQ(handle_info.rank, 0);
  ASSERT_EQ(handle_info.shape[0], 0);

  ArgInfo vec_info;
  ArgOverride<std::vector<int8_t>>::apply(vec_info);
  ASSERT_EQ(vec_info.type, GXF_PARAMETER_TYPE_INT8);
  ASSERT_EQ(vec_info.type_name, "Int8");
  ASSERT_EQ(vec_info.rank, 1);
  ASSERT_EQ(vec_info.shape[0], -1);

  ArgInfo vec_handle_info;
  ArgOverride<std::vector<Handle<Allocator>>>::apply(vec_handle_info);
  ASSERT_EQ(vec_handle_info.type, GXF_PARAMETER_TYPE_HANDLE);
  ASSERT_EQ(vec_handle_info.type_name, "nvidia::gxf::Allocator");
  ASSERT_EQ(vec_handle_info.rank, 1);
  ASSERT_EQ(vec_handle_info.shape[0], -1);

  ArgInfo vec_vec_info;
  ArgOverride<std::vector<std::vector<uint8_t>>>::apply(vec_handle_info);
  ASSERT_EQ(vec_handle_info.type, GXF_PARAMETER_TYPE_UINT8);
  ASSERT_EQ(vec_handle_info.type_name, "UInt8");
  ASSERT_EQ(vec_handle_info.rank, 2);
  ASSERT_EQ(vec_handle_info.shape[0], -1);
  ASSERT_EQ(vec_handle_info.shape[1], -1);

  ArgInfo array_info;
  ArgOverride<std::array<int8_t, 3>>::apply(array_info);
  ASSERT_EQ(array_info.type, GXF_PARAMETER_TYPE_INT8);
  ASSERT_EQ(array_info.type_name, "Int8");
  ASSERT_EQ(array_info.rank, 1);
  ASSERT_EQ(array_info.shape[0], 3);

  ArgInfo array_handle_info;
  ArgOverride<std::array<Handle<Allocator>, 3>>::apply(array_handle_info);
  ASSERT_EQ(array_handle_info.type, GXF_PARAMETER_TYPE_HANDLE);
  ASSERT_EQ(array_handle_info.type_name, "nvidia::gxf::Allocator");
  ASSERT_EQ(array_handle_info.rank, 1);
  ASSERT_EQ(array_handle_info.shape[0], 3);

  ArgInfo arr_arr_info;
  ArgOverride<std::array<std::array<uint8_t, 3>, 3>>::apply(arr_arr_info);
  ASSERT_EQ(arr_arr_info.type, GXF_PARAMETER_TYPE_UINT8);
  ASSERT_EQ(arr_arr_info.type_name, "UInt8");
  ASSERT_EQ(arr_arr_info.rank, 2);
  ASSERT_EQ(arr_arr_info.shape[0], 3);
  ASSERT_EQ(arr_arr_info.shape[1], 3);

  ArgInfo complex64_info;
  ArgOverride<std::complex<float>>::apply(complex64_info);
  ASSERT_EQ(complex64_info.type, GXF_PARAMETER_TYPE_COMPLEX64);
  ASSERT_EQ(complex64_info.type_name, "Complex64");
  ASSERT_EQ(complex64_info.rank, 0);

  ArgInfo complex128_info;
  ArgOverride<std::complex<double>>::apply(complex128_info);
  ASSERT_EQ(complex128_info.type, GXF_PARAMETER_TYPE_COMPLEX128);
  ASSERT_EQ(complex128_info.type_name, "Complex128");
  ASSERT_EQ(complex128_info.rank, 0);
}

TEST_F(GxfArg, ProxyComponent) {
  auto tx1 = ProxyComponent("nvidia::gxf::DoubleBufferTransmitter", "tx1", {Arg("capacity", 10)});

  Arg copy_construct{"CopyConstruct", tx1};
  ASSERT_TRUE(copy_construct.has_value());
  ASSERT_EQ(std::any_cast<ProxyComponent>(copy_construct.value()).type_name(), "nvidia::gxf::DoubleBufferTransmitter");
  ASSERT_EQ(copy_construct.parameter_type(), GXF_PARAMETER_TYPE_HANDLE);
  ASSERT_EQ(copy_construct.handle_uid(), kUnspecifiedUid);
  ASSERT_EQ(copy_construct.handle_tid(), GxfTidNull());
  ASSERT_EQ(copy_construct.arg_info().rank, 0);
  ASSERT_EQ(copy_construct.arg_info().shape[0], 0);

  Arg copy_assign("CopyAssign");
  copy_assign = tx1;
  ASSERT_TRUE(copy_assign.has_value());
  ASSERT_EQ(std::any_cast<ProxyComponent>(copy_assign.value()).type_name(), "nvidia::gxf::DoubleBufferTransmitter");
  ASSERT_EQ(copy_assign.parameter_type(), GXF_PARAMETER_TYPE_HANDLE);
  ASSERT_EQ(copy_assign.handle_uid(), kUnspecifiedUid);
  ASSERT_EQ(copy_assign.handle_tid(), GxfTidNull());
  ASSERT_EQ(copy_assign.arg_info().rank, 0);
  ASSERT_EQ(copy_assign.arg_info().shape[0], 0);

  auto tx2 = ProxyComponent("nvidia::gxf::DoubleBufferTransmitter", "tx2", {Arg("capacity", 10)});
  Arg move_construct("MoveConstruct", std::move(tx2));
  ASSERT_TRUE(move_construct.has_value());
  ASSERT_EQ(std::any_cast<ProxyComponent>(move_construct.value()).type_name(), "nvidia::gxf::DoubleBufferTransmitter");
  ASSERT_EQ(move_construct.parameter_type(), GXF_PARAMETER_TYPE_HANDLE);
  ASSERT_EQ(move_construct.handle_uid(), kUnspecifiedUid);
  ASSERT_EQ(move_construct.handle_tid(), GxfTidNull());
  ASSERT_EQ(move_construct.arg_info().rank, 0);
  ASSERT_EQ(move_construct.arg_info().shape[0], 0);

  auto tx3 = ProxyComponent("nvidia::gxf::DoubleBufferTransmitter", "tx3", {Arg("capacity", 10)});
  Arg move_assign("MoveAssign");
  move_assign = std::move(tx3);
  ASSERT_TRUE(move_assign.has_value());
  ASSERT_EQ(std::any_cast<ProxyComponent>(move_assign.value()).type_name(), "nvidia::gxf::DoubleBufferTransmitter");
  ASSERT_EQ(move_assign.parameter_type(), GXF_PARAMETER_TYPE_HANDLE);
  ASSERT_EQ(move_assign.handle_uid(), kUnspecifiedUid);
  ASSERT_EQ(move_assign.handle_tid(), GxfTidNull());
  ASSERT_EQ(move_assign.arg_info().rank, 0);
  ASSERT_EQ(move_assign.arg_info().shape[0], 0);
}

TEST_F(GxfArg, Handle) {
  auto maybe_entity = Entity::New(context);
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  auto maybe_tx = entity.add<DoubleBufferTransmitter>("Tx");
  ASSERT_TRUE(maybe_tx);
  Handle<DoubleBufferTransmitter> tx1 = maybe_tx.value();

  Arg copy_construct{"CopyConstruct", tx1};
  YAML::Emitter copy_out;
  ASSERT_TRUE(copy_construct.has_value());
  ASSERT_EQ(std::any_cast<Handle<DoubleBufferTransmitter>>(copy_construct.value()), tx1);
  ASSERT_EQ(copy_construct.parameter_type(), GXF_PARAMETER_TYPE_HANDLE);
  ASSERT_NE(copy_construct.handle_uid(), kNullUid);
  ASSERT_EQ(copy_construct.handle_tid(), DoubleBufferTxTID);
  ASSERT_EQ(copy_construct.arg_type_name(), "nvidia::gxf::DoubleBufferTransmitter");
  ASSERT_EQ(copy_construct.arg_info().rank, 0);
  ASSERT_EQ(copy_construct.arg_info().shape[0], 0);
  copy_out << copy_construct.yaml_node();
  ASSERT_EQ(std::string(copy_out.c_str()), std::string("__entity_2/Tx"));

  Arg copy_assign("CopyAssign");
  copy_assign = tx1;
  ASSERT_TRUE(copy_assign.has_value());
  ASSERT_EQ(std::any_cast<Handle<DoubleBufferTransmitter>>(copy_assign.value()), tx1);
  ASSERT_EQ(copy_assign.parameter_type(), GXF_PARAMETER_TYPE_HANDLE);
  ASSERT_NE(copy_assign.handle_uid(), kNullUid);
  ASSERT_EQ(copy_assign.handle_tid(), DoubleBufferTxTID);
  ASSERT_EQ(copy_assign.arg_type_name(), "nvidia::gxf::DoubleBufferTransmitter");
  ASSERT_EQ(copy_assign.arg_info().rank, 0);
  ASSERT_EQ(copy_assign.arg_info().shape[0], 0);

  auto maybe_tx2 = entity.add<DoubleBufferTransmitter>("Tx2");
  ASSERT_TRUE(maybe_tx2);
  Handle<DoubleBufferTransmitter> tx2 = maybe_tx2.value();
  Arg move_construct("MoveConstruct", std::move(tx2));
  ASSERT_TRUE(move_construct.has_value());
  // ASSERT_EQ(std::any_cast<Handle<DoubleBufferTransmitter>>(tx_handle.value()), tx1);
  ASSERT_EQ(move_construct.parameter_type(), GXF_PARAMETER_TYPE_HANDLE);
  ASSERT_NE(move_construct.handle_uid(), kNullUid);
  ASSERT_EQ(move_construct.handle_tid(), DoubleBufferTxTID);
  ASSERT_EQ(move_construct.arg_type_name(), "nvidia::gxf::DoubleBufferTransmitter");
  ASSERT_EQ(move_construct.arg_info().rank, 0);
  ASSERT_EQ(move_construct.arg_info().shape[0], 0);

  auto maybe_tx3 = entity.add<DoubleBufferTransmitter>("Tx3");
  ASSERT_TRUE(maybe_tx3);
  Handle<DoubleBufferTransmitter> tx3 = maybe_tx3.value();
  Arg move_assign("MoveAssign");
  move_assign = std::move(tx3);
  ASSERT_TRUE(move_assign.has_value());
  // ASSERT_EQ(std::any_cast<Handle<DoubleBufferTransmitter>>(tx_handle.value()), tx3);
  ASSERT_EQ(move_assign.parameter_type(), GXF_PARAMETER_TYPE_HANDLE);
  ASSERT_NE(move_assign.handle_uid(), kNullUid);
  ASSERT_EQ(move_assign.handle_tid(), DoubleBufferTxTID);
  ASSERT_EQ(move_assign.arg_type_name(), "nvidia::gxf::DoubleBufferTransmitter");
  ASSERT_EQ(move_assign.arg_info().rank, 0);
  ASSERT_EQ(move_assign.arg_info().shape[0], 0);
}

TEST_F(GxfArg, Vector) {
  std::vector<int8_t> simple_int{1, 2, 3};
  Arg simple_vector = Arg("Int", simple_int);
  ASSERT_TRUE(simple_vector.has_value());
  GXF_ASSERT_EQ(std::any_cast<std::vector<int8_t>>(simple_vector.value())[0], 1);
  GXF_ASSERT_EQ(std::any_cast<std::vector<int8_t>>(simple_vector.value())[1], 2);
  GXF_ASSERT_EQ(std::any_cast<std::vector<int8_t>>(simple_vector.value())[2], 3);

  auto maybe_entity = Entity::New(context);
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  auto maybe_block = entity.add<BlockMemoryPool>("Block");
  ASSERT_TRUE(maybe_block);
  auto block = maybe_block.value();

  auto maybe_unbounded = entity.add<UnboundedAllocator>("Unbounded");
  ASSERT_TRUE(maybe_unbounded);
  auto unbounded = maybe_unbounded.value();

  Arg allocator_list = Arg("Alloc");
  std::vector<Handle<Allocator>> allocator_vector{block, unbounded};
  allocator_list = allocator_vector;
  ASSERT_TRUE(allocator_list.has_value());
  ASSERT_EQ(allocator_list.arg_type_name(), "nvidia::gxf::Allocator");
  ASSERT_EQ(allocator_list.arg_info().rank, 1);
  ASSERT_EQ(allocator_list.arg_info().shape[0], -1);
}

TEST_F(GxfArg, AllParameterTypes) {
  std::vector<Arg> all_args{
      Arg("bool", true),
      Arg("int8_t", int8_t(INT8_MAX)),
      Arg("int16_t", int16_t(INT16_MAX)),
      Arg("int32_t", int32_t(INT32_MAX)),
      Arg("int64_t", int64_t(INT64_MAX)),
      Arg("uint8_t", uint8_t(UINT8_MAX)),
      Arg("uint16_t", uint16_t(UINT16_MAX)),
      Arg("uint32_t", uint32_t(UINT32_MAX)),
      Arg("uint64_t", uint64_t(UINT64_MAX)),
      Arg("float", float(FLT_MAX)),
      Arg("double", double(DBL_MAX)),
      Arg("complex64", std::complex<float>(7.5, 3.0)),
      Arg("complex128", std::complex<double>(1.234, 5.678)),
      Arg("std::string", std::string("string")),
      Arg("Handle<Allocator>", Handle<Allocator>::Null()),
      Arg("std::array<bool, 2>", std::array<bool, 2>{true, false}),
      Arg("std::array<int8_t, 2>", std::array<int8_t, 2>{INT8_MAX, INT8_MIN}),
      Arg("std::array<int16_t, 2>", std::array<int16_t, 2>{INT16_MAX, INT16_MIN}),
      Arg("std::array<int32_t, 2>", std::array<int32_t, 2>{INT32_MAX, INT32_MIN}),
      Arg("std::array<int64_t, 2>", std::array<int64_t, 2>{INT64_MAX, INT64_MIN}),
      Arg("std::array<uint8_t, 2>", std::array<uint8_t, 2>{UINT8_MAX, 0}),
      Arg("std::array<uint16_t, 2>", std::array<uint16_t, 2>{UINT16_MAX, 0}),
      Arg("std::array<uint32_t, 2>", std::array<uint32_t, 2>{UINT32_MAX, 0}),
      Arg("std::array<uint64_t, 2>", std::array<uint64_t, 2>{UINT64_MAX, 0}),
      Arg("std::array<float, 2>", std::array<float, 2>{FLT_MAX, FLT_MIN}),
      Arg("std::array<double, 2>", std::array<double, 2>{DBL_MAX, DBL_MIN}),
      Arg("std::array<Handle<Allocator>, 1>",
          std::array<Handle<Allocator>, 1>{Handle<Allocator>::Null()}),
      Arg("std::array<std::string, 1>", std::array<std::string, 1>{std::string("string")}),
      Arg("std::vector<bool>", std::vector<bool>(true, false)),
      Arg("std::vector<int8_t>", std::vector<int8_t>{INT8_MAX, INT8_MIN}),
      Arg("std::vector<int16_t>", std::vector<int16_t>{INT16_MAX, INT16_MIN}),
      Arg("std::vector<int32_t>", std::vector<int32_t>{INT32_MAX, INT32_MIN}),
      Arg("std::vector<int64_t>", std::vector<int64_t>{INT64_MAX, INT64_MIN}),
      Arg("std::vector<uint8_t>", std::vector<uint8_t>{UINT8_MAX, 0}),
      Arg("std::vector<uint16_t>", std::vector<uint16_t>{UINT16_MAX, 0}),
      Arg("std::vector<uint32_t>", std::vector<uint32_t>{UINT32_MAX, 0}),
      Arg("std::vector<uint64_t>", std::vector<uint64_t>{UINT64_MAX, 0}),
      Arg("std::vector<float>", std::vector<float>{FLT_MAX, FLT_MIN}),
      Arg("std::vector<double>", std::vector<double>{DBL_MAX, DBL_MIN}),
      Arg("std::vector<Handle<Allocator>>",
          std::vector<Handle<Allocator>>{Handle<Allocator>::Null()}),
      Arg("std::vector<std::string>", std::vector<std::string>{std::string("string")}),
      Arg("std::vector<std::vector<bool>>", std::vector<std::vector<bool>>{{true, false}}),
      Arg("std::vector<std::vector<int8_t>>", std::vector<std::vector<int8_t>>()),
      Arg("std::vector<std::vector<int16_t>>", std::vector<std::vector<int16_t>>()),
      Arg("std::vector<std::vector<int32_t>>", std::vector<std::vector<int32_t>>()),
      Arg("std::vector<std::vector<int64_t>>", std::vector<std::vector<int64_t>>()),
      Arg("std::vector<std::vector<uint8_t>>", std::vector<std::vector<uint8_t>>()),
      Arg("std::vector<std::vector<uint16_t>>", std::vector<std::vector<uint16_t>>()),
      Arg("std::vector<std::vector<uint32_t>>", std::vector<std::vector<uint32_t>>()),
      Arg("std::vector<std::vector<uint64_t>>", std::vector<std::vector<uint64_t>>()),
      Arg("std::vector<std::vector<float>>", std::vector<std::vector<float>>()),
      Arg("std::vector<std::vector<double>>", std::vector<std::vector<double>>()),
      Arg("std::vector<std::vector<std::string>>", std::vector<std::vector<std::string>>()),
  };

  for (auto& arg : all_args) {
    GXF_LOG_INFO("Typename %s", arg.arg_type_name().c_str());
    // ASSERT_EQ(arg.key(), arg.arg_type_name());
    ASSERT_NE(arg.arg_type_name().c_str(), "(custom)");
    ASSERT_NE(arg.parameter_type(), GXF_PARAMETER_TYPE_CUSTOM);
  }
}

TEST_F(GxfArg, ArgParse) {

  auto maybe_entity = Entity::New(context);
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  auto maybe_tx = entity.add<DoubleBufferTransmitter>("Tx");
  ASSERT_TRUE(maybe_tx);
  Handle<DoubleBufferTransmitter> tx = maybe_tx.value();

  Arg tx_arg("out", tx);

  auto maybe_rx = entity.add<DoubleBufferReceiver>("Rx");
  ASSERT_TRUE(maybe_rx);
  Handle<DoubleBufferReceiver> rx = maybe_rx.value();
  Arg rx_arg("in", rx);

  auto maybe_forward = entity.add<Forward>("Forward");
  ASSERT_TRUE(maybe_forward);
  Handle<Forward> forward = maybe_forward.value();

  ASSERT_TRUE(applyArg(forward, tx_arg));
  ASSERT_TRUE(applyArg(forward, rx_arg));
}

TEST_F(GxfArg, FindParse) {
  auto maybe_entity = Entity::New(context);
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  auto maybe_tx = entity.add<DoubleBufferTransmitter>("Tx");
  ASSERT_TRUE(maybe_tx);
  Handle<DoubleBufferTransmitter> tx = maybe_tx.value();

  Arg tx_arg("out", tx);

  auto maybe_rx = entity.add<DoubleBufferReceiver>("Rx");
  ASSERT_TRUE(maybe_rx);
  Handle<DoubleBufferReceiver> rx = maybe_rx.value();
  Arg rx_arg("in", rx);

  auto maybe_forward = entity.add<Forward>("Forward");
  ASSERT_TRUE(maybe_forward);
  Handle<Forward> forward = maybe_forward.value();

  ASSERT_TRUE(applyArg(forward, tx_arg));
  ASSERT_TRUE(applyArg(forward, rx_arg));

  const auto foundArg = findArg({tx_arg, rx_arg}, "in", GXF_PARAMETER_TYPE_HANDLE);
  ASSERT_TRUE(foundArg.has_value());

  const auto invalidArg = findArg({tx_arg, rx_arg}, "invalid", GXF_PARAMETER_TYPE_STRING);
  ASSERT_FALSE(invalidArg.has_value());

  Arg arg("in");
  ASSERT_FALSE(applyArg(forward, arg));
}

TEST_F(GxfArg, ApplyArgWithoutValue) {
  auto maybe_entity = Entity::New(context);
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  auto maybe_forward = entity.add<Forward>("Forward");
  ASSERT_TRUE(maybe_forward);
  Handle<Forward> forward = maybe_forward.value();

  Arg arg("in");
  ASSERT_FALSE(applyArg(forward, arg));
}

TEST_F(GxfArg, ArgAPI) {
  auto maybe_entity = Entity::New(context);
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  auto maybe_tx = entity.add<DoubleBufferTransmitter>("Tx");
  ASSERT_TRUE(maybe_tx);
  Handle<DoubleBufferTransmitter> tx = maybe_tx.value();
  auto backSize = tx->back_size_abi();
  ASSERT_EQ(backSize, (size_t)0);
  ASSERT_TRUE(tx->pop_io_abi(nullptr) == GXF_SUCCESS);
  ASSERT_TRUE(tx->peek_abi(nullptr, 0) == GXF_ARGUMENT_NULL);

  std::unique_ptr<gxf_uid_t> uid = std::make_unique<gxf_uid_t>();
  ASSERT_TRUE(tx->peek_abi(uid.get(), 0) == GXF_FAILURE);
}

}  // namespace gxf
}  // namespace nvidia
