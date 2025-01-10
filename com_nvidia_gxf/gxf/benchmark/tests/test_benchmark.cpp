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
#include "gxf/benchmark/tests/async_trigger_data_replayer.hpp"
#include "gxf/benchmark/tests/benchmark_report_checker.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0xfffbb93f01dd41d1, 0xa2a695c4530b044e, "TestBenchmarkExtension",
                         "Helper extension with components used to test benchmark components",
                         "NVIDIA", "0.0.1", "NVIDIA");

GXF_EXT_FACTORY_ADD(0x4271c7389f1a45c4, 0xb6babfe2324ec1f0,
                    nvidia::gxf::benchmark::test::AsyncTriggerDataReplayer,
                    nvidia::gxf::Codelet,
                    "Triggers AsynchronousSchedulingTerm based on data replay commands");

GXF_EXT_FACTORY_ADD(0x2826699dcbcf49cb, 0x98ef77f740dd27c6,
                    nvidia::gxf::benchmark::test::BenchmarkReportChecker,
                    nvidia::gxf::Codelet,
                    "Checks if a generated benchmark report is valid");

GXF_EXT_FACTORY_END()
