/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "gxf/std/extension_factory_helper.hpp"
#include "gxf/test/components/camera_message_generator.hpp"
#include "gxf/test/components/entity_monitor.hpp"
#include "gxf/test/components/mock_allocator.hpp"
#include "gxf/test/components/mock_codelet.hpp"
#include "gxf/test/components/mock_failure.hpp"
#include "gxf/test/components/mock_receiver.hpp"
#include "gxf/test/components/mock_transmitter.hpp"
#include "gxf/test/components/tensor_comparator.hpp"
#include "gxf/test/components/tensor_generator.hpp"
#include "gxf/test/extensions/test_helpers.hpp"
#include "gxf/test/extensions/test_metric.hpp"
#include "gxf/test/extensions/test_parameters.hpp"

GXF_EXT_FACTORY_BEGIN()
  GXF_EXT_FACTORY_SET_INFO(0x1b99ffebc2504ced, 0x811762ac05969a50, "TestHelperExtension",
                           "Helper extension with components used to test std components",
                           "NVIDIA", "2.3.0", "NVIDIA");
  GXF_EXT_FACTORY_SET_DISPLAY_INFO("Test Helper Extension", "Test", "GXF Test Helper Extension");
  GXF_EXT_FACTORY_ADD(0x77d3cb561ddb4656, 0x828a3955ad78fb27,
                       nvidia::gxf::test::LoadParameterFromYamlTest, nvidia::gxf::Component,
                       "Tests that parameters are loaded correctly from yaml");
  GXF_EXT_FACTORY_ADD(0xe9234c1ad5f8445c, 0xae9118bcda197032,
                       nvidia::gxf::test::RegisterParameterInterfaceTest, nvidia::gxf::Component,
                       "Tests all variations of the parameter interface");
  GXF_EXT_FACTORY_ADD(0xf6a9bda17a0d1642, 0xc2d06c87c645218f,
                       nvidia::gxf::test::FixedVectorParameterTest, nvidia::gxf::Component,
                       "Tests parameter support for nvidia::FixedVector");
  GXF_EXT_FACTORY_ADD(0x405d8e062d3f45f1, 0x84c492ac9ef3c67c,
                       nvidia::gxf::test::StdParameterTest, nvidia::gxf::Component,
                       "Tests parameter support for STD types like std::vector");
  GXF_EXT_FACTORY_ADD(0x92f49e50aded48a8, 0x8b97a274e227875f,
                       nvidia::gxf::test::TestHandleParameter, nvidia::gxf::Component,
                       "Tests getting parameters using try_get");
  GXF_EXT_FACTORY_ADD(0xfa56c02660824b44, 0x9fc75efd1e1d6a13,
                      nvidia::gxf::test::TestGxfParameterSetFromYamlNode, nvidia::gxf::Component,
                      "Tests various parameters with GxfParameterSetFromYamlNode() API");
  GXF_EXT_FACTORY_ADD(0xc076cb7155ca4687, 0x8c84c789818f5cf7,
                       nvidia::gxf::test::StepCount, nvidia::gxf::Codelet,
                       "Tests that the codelet was stepped a certain number of times");
  GXF_EXT_FACTORY_ADD(0xa4f26a81a7dc4cdb, 0x8ad9c638c2de48f8,
                       nvidia::gxf::test::AsyncPingRx, nvidia::gxf::Codelet,
                       "Receives an entity at async time intervals");
  GXF_EXT_FACTORY_ADD(0x64866edc725b11eb, 0x9dfa37dfe2305968,
                       nvidia::gxf::test::PingPollRx, nvidia::gxf::Codelet,
                       "Polls and receives an entity and check counter on stop");
  GXF_EXT_FACTORY_ADD(0x85058d3a553911eb, 0xbe0b873d2618d774,
                       nvidia::gxf::test::PingBatchRx, nvidia::gxf::Codelet,
                       "Receives an entity");
  GXF_EXT_FACTORY_ADD(0x8b2e8b0f02364d5f, 0x3e572048949a43ef,
                       nvidia::gxf::test::ScheduledPingTx, nvidia::gxf::Codelet,
                       "Sends an entity at differing time intervals");
  GXF_EXT_FACTORY_ADD(0x1c211f155e5a4efc, 0x8daa19f351d19ad6,
                       nvidia::gxf::test::AsyncPingTx, nvidia::gxf::Codelet,
                       "Sends an entity at async time intervals");
  GXF_EXT_FACTORY_ADD(0xebe83f933e8e4c09, 0x8c124ab7390ac565,
                       nvidia::gxf::test::Generator, nvidia::gxf::Codelet,
                       "Generates integer and fibonacci messages");
  GXF_EXT_FACTORY_ADD(0x6019517f60844d1b, 0x9eff178d6891e9e8,
                       nvidia::gxf::test::Pop, nvidia::gxf::Codelet,
                       "Pops an incoming message at the receiver");
  GXF_EXT_FACTORY_ADD(0x15ac607f553942e0, 0xb1f09b841e1f9ecd,
                       nvidia::gxf::test::Print, nvidia::gxf::Codelet,
                       "Receives a message with tensors and prints the element count to debug log");
  GXF_EXT_FACTORY_ADD(0xc73d835ebf5b4a54, 0x9cd5370e1256b650,
                       nvidia::gxf::test::PrintTensor, nvidia::gxf::Codelet,
                       "Receives a message with tensors and prints them to the console.");
  GXF_EXT_FACTORY_ADD(0x2ca901d00d384a06, 0x951edc0d655ac5a6,
                       nvidia::gxf::test::ArbitraryPrecisionFactorial, nvidia::gxf::Codelet,
                       "Generates an arbitrary precision factorial message");
  GXF_EXT_FACTORY_ADD(0x2ed6049cc6ad4d89, 0x8e28078be6776adf,
                       nvidia::gxf::test::IntegerSinSum, nvidia::gxf::Codelet,
                       "Creates a tensor message with a cumulative sum of sines");
  GXF_EXT_FACTORY_ADD(0xd2abde61ff76ea8b, 0xfdc74e01d70259af,
                      nvidia::gxf::test::TensorGenerator, nvidia::gxf::Codelet,
                      "Generates random tensors with a timestamp");
  GXF_EXT_FACTORY_ADD(0xe99f8fe0c7ab9674, 0xc4ef5778874b7c3f,
                      nvidia::gxf::test::TensorComparator, nvidia::gxf::Codelet,
                      "Compares two tensor messages for equality");
  GXF_EXT_FACTORY_ADD(0x92002082602311eb, 0xaeea03cedbfc916f,
                       nvidia::gxf::test::TestTensorStrides, nvidia::gxf::Codelet,
                       "Tests Tensor Stride");
  GXF_EXT_FACTORY_ADD(0x557d64785a864052, 0xa2acf94edb0341ca,
                       nvidia::gxf::test::TestMetricLogger, nvidia::gxf::Codelet,
                       "Tests Metric component");
  GXF_EXT_FACTORY_ADD(0xd3f52056b8a0974d, 0xf8d3c321bdf247ee,
                       nvidia::gxf::test::EntityMonitor, nvidia::gxf::Monitor,
                       "Monitors entities during runtime and logs status to console.");
  GXF_EXT_FACTORY_ADD(0xb6c6697a3b2d79fa, 0x0bb8c8d6ad0f92ac,
                       nvidia::gxf::test::VectorParameterTest, nvidia::gxf::Component,
                       "Tests that vector params are loaded correctly from yaml and get/set works");
  GXF_EXT_FACTORY_ADD(0x4f7b92fcfb4d1096, 0xeebf2cf958e74c45,
                      nvidia::gxf::test::TestTimestampTx, nvidia::gxf::Codelet,
                      "Tests the pubtime and acqtime");
  GXF_EXT_FACTORY_ADD(0xa0204781cabf69d7, 0x56153ac8398984ca,
                      nvidia::gxf::test::TestTimestampRx, nvidia::gxf::Codelet,
                      "Tests the pubtime and acqtime");
  GXF_EXT_FACTORY_ADD(0x121178a5ee024251, 0xb2d30663306bf3c3,
                      nvidia::gxf::test::TestLogger, nvidia::gxf::Codelet,
                      "Prints sample logs for various log levels");
  GXF_EXT_FACTORY_ADD(0xbb138e69066aba28, 0xd392c98a9b0849b4,
                      nvidia::gxf::test::PeriodicSchedulingTermWithDelay,
                      nvidia::gxf::PeriodicSchedulingTerm,
                      "Adds a random delay to the periodic scheduling term");
  GXF_EXT_FACTORY_ADD(0x93585693d816b6c5, 0xcb06ba6ec534dfd4,
                      nvidia::gxf::test::MockAllocator, nvidia::gxf::Allocator,
                      "Memory allocator used to facilitate testing");
  GXF_EXT_FACTORY_ADD(0x9fa09605848581ab, 0x92c23096508cc4a1,
                      nvidia::gxf::test::MockTransmitter, nvidia::gxf::Transmitter,
                      "Entity transmitter used to facilitate testing");
  GXF_EXT_FACTORY_ADD(0xe9c70ddbf2235546, 0xbb7b118048380c21,
                      nvidia::gxf::test::MockReceiver, nvidia::gxf::Receiver,
                      "Entity receiver used to facilitate testing");
  GXF_EXT_FACTORY_ADD(0x85ed3ce306114314, 0x94dfda4f507f75a0,
                      nvidia::gxf::test::TestConfigurationSet, nvidia::gxf::Codelet,
                      "Tests that verifies the runtime parameter change");
  GXF_EXT_FACTORY_ADD(0xab79d2a73f474f97, 0xb70533463800e379,
                      nvidia::gxf::test::TestCameraMessage, nvidia::gxf::Codelet,
                      "Tests that verifies consistent mechanism to pass camera images");
  GXF_EXT_FACTORY_ADD(0xa055fbf8758811ed, 0xa423331e2d38b116,
                      nvidia::gxf::test::WaitSchedulingTerm,
                      nvidia::gxf::SchedulingTerm,
                      "Scheduling term that is always in WAIT state");
  GXF_EXT_FACTORY_ADD(0x263ba11abf9d11ed, 0xa3ed8f40163997e9, nvidia::gxf::test::MockFailure,
                      nvidia::gxf::Codelet,
                      "A component that will return failure if ticked more than expected");
  GXF_EXT_FACTORY_ADD(0x01401bae1cf64e91, 0xadd00c88b988b04c,
                    nvidia::gxf::test::MockCodelet, nvidia::gxf::Codelet,
                    "Mock generic codelet that receives message, execute, then transmit message");
  GXF_EXT_FACTORY_ADD(0xcd944caff32f46cc, 0xa8a744977a4805a6,
                    nvidia::gxf::test::Frame, nvidia::gxf::Component,
                    "Test frame that contains timestamp and frame index");
  GXF_EXT_FACTORY_END()
