/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gxf/benchmark/allocator_metrics_calculator.hpp"
#include "gxf/benchmark/basic_metrics_calculator.hpp"
#include "gxf/benchmark/benchmark_allocator.hpp"
#include "gxf/benchmark/benchmark_allocator_sink.hpp"
#include "gxf/benchmark/benchmark_controller.hpp"
#include "gxf/benchmark/benchmark_publisher.hpp"
#include "gxf/benchmark/benchmark_sink.hpp"
#include "gxf/benchmark/dummy_message_generator.hpp"
#include "gxf/benchmark/entity_buffer.hpp"
#include "gxf/benchmark/gems/data_replay_control.hpp"
#include "gxf/benchmark/performance_calculator_base.hpp"
#include "gxf/benchmark/resource_profiler.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0xb53ebe0ce4274485, 0x8ca2a4ddc8664bec,
                         "BenchmarkExtension",
                         "Extension for benchmarking GXF components and graphs",
                         "NVIDIA", "0.0.1", "NVIDIA");

GXF_EXT_FACTORY_SET_DISPLAY_INFO("Benchmark Extension", "Benchmark", "GXF Benchmark Extension");

GXF_EXT_FACTORY_ADD(0xd7adb7a6d0954e3d, 0x843959a0d7c8f69e,
                    nvidia::gxf::benchmark::PerformanceCalculatorBase,
                    nvidia::gxf::Component,
                    "Base class of performance calculators");

GXF_EXT_FACTORY_ADD(0x135cc6029a18437b, 0x8d1dd4e0a1fbd45d,
                    nvidia::gxf::benchmark::BasicMetricsCalculator,
                    nvidia::gxf::benchmark::PerformanceCalculatorBase,
                    "Calculates performance outcomes for basic metrics");

GXF_EXT_FACTORY_ADD(0xf99e06c2e8b649cf, 0xb7b0556185b7d274,
                    nvidia::gxf::benchmark::ResourceProfilerBase,
                    nvidia::gxf::benchmark::PerformanceCalculatorBase,
                    "Base class of resource profilers");

GXF_EXT_FACTORY_ADD(0x7a22ea90af3f4b2f, 0x9a84cbcae845eef2,
                    nvidia::gxf::benchmark::AllocatorMetricsCalculator,
                    nvidia::gxf::benchmark::PerformanceCalculatorBase,
                    "Calculates performance outcomes for allocator metrics");

GXF_EXT_FACTORY_ADD(0x1b7147ba6f28449a, 0xacdb74f980c976b9,
                    nvidia::gxf::benchmark::ResourceProfiler,
                    nvidia::gxf::benchmark::ResourceProfilerBase,
                    "Profiles resource utilizations");

GXF_EXT_FACTORY_ADD(0xa2e06c9967d14f24, 0xa393dd3e0fb76242,
                    nvidia::gxf::benchmark::EntityBuffer,
                    nvidia::gxf::Codelet,
                    "Buffers incoming message entities");

GXF_EXT_FACTORY_ADD(0x52d9e7c72a394dd6, 0x9b10f6242dc4c6d6,
                    nvidia::gxf::benchmark::BenchmarkPublisher,
                    nvidia::gxf::Codelet,
                    "Publishes buffered benchmark messages");


GXF_EXT_FACTORY_ADD(0x612ec0fce0344c83, 0xa3a72f1e9826a1fb,
                    nvidia::gxf::benchmark::BenchmarkSinkBase, nvidia::gxf::Codelet,
                    "Benchmark sink base type");

GXF_EXT_FACTORY_ADD(0x35885a99e29a4cdb, 0xa08d7b288ccf41f1,
                    nvidia::gxf::benchmark::BenchmarkSink,
                    nvidia::gxf::benchmark::BenchmarkSinkBase,
                    "Records message arrival timestamps");

GXF_EXT_FACTORY_ADD_0(0xfd1121db302a4b78, 0x8fa00b246a5c1950,
                      nvidia::gxf::benchmark::DataReplayControl,
                      "Message component used to control data replay during runtime");

GXF_EXT_FACTORY_ADD(0x763b55415b654b84, 0xa6ee837d9459e903,
                    nvidia::gxf::benchmark::BenchmarkController,
                    nvidia::gxf::Codelet,
                    "Controls benchmark flow");

GXF_EXT_FACTORY_ADD(0xdfeb6b8bd7d94350, 0xa8cb1a7ea40825de,
                    nvidia::gxf::benchmark::DummyMessageGenerator, nvidia::gxf::Codelet,
                    "Generates dummy messages");

GXF_EXT_FACTORY_ADD(0x2c9a44d363f840f0, 0x9225e87e6fd1679d,
                    nvidia::gxf::benchmark::BenchmarkAllocatorSink,
                    nvidia::gxf::benchmark::BenchmarkSinkBase,
                    "Records allocate/free duration for allocator component");

GXF_EXT_FACTORY_ADD(0xf559d7ccbdf449b6, 0x8ec0a7b6ecb4b222,
                    nvidia::gxf::benchmark::BenchmarkAllocator,
                    nvidia::gxf::Codelet,
                    "Benchmarks allocator component");

GXF_EXT_FACTORY_END()
