/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gxf/stream/tests/test_gxf_stream_sync_cuda_helper.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0x31bee146e13d4161, 0x8d6a37403048d618, "TestGxfStreamExtension",
                         "Testing Gxf Stream related components", "Nvidia_Gxf", "0.5.0", "NVIDIA");

GXF_EXT_FACTORY_SET_DISPLAY_INFO("Test Stream Extension", "Stream", "GXF Test Stream Extension");

GXF_EXT_FACTORY_ADD(0x071fab5316724442, 0x87d0a104b90836e2,
                    nvidia::gxf::stream::test::StreamBasedOpsNew,
                    nvidia::gxf::Codelet, "A base class of cuda streams ops");

GXF_EXT_FACTORY_ADD(0x7d91f809494e4049, 0x992daecc406796eb,
                    nvidia::gxf::stream::test::StreamTensorGeneratorNew,
                    nvidia::gxf::stream::test::StreamBasedOpsNew,
                    "Generates cuda tensors with the stream id along with GXF StreamSync");

GXF_EXT_FACTORY_ADD(0xaf1edd6cc3bf4558, 0xbd02706dfc98fac9,
                    nvidia::gxf::stream::test::CublasDotProductNew,
                    nvidia::gxf::stream::test::StreamBasedOpsNew,
                    "Cublas dot product calculation");

GXF_EXT_FACTORY_ADD(0x4a367bcf3ea04488, 0x88cd571060119d43,
                    nvidia::gxf::stream::test::HostDotProductNew,
                    nvidia::gxf::Codelet, "CPU dot product calculation");

GXF_EXT_FACTORY_ADD(0xc6f7de7f978749ce, 0x9e2a52464d027a4c,
                    nvidia::gxf::stream::test::MemCpy2HostNew,
                    nvidia::gxf::stream::test::StreamBasedOpsNew, "Mem copy device to host async");

GXF_EXT_FACTORY_ADD(0xc9d7381d79b241a1, 0x83025e190a84d53b,
                    nvidia::gxf::stream::test::VerifyEqualNew,
                    nvidia::gxf::Codelet, "Verify whether 2 input tensors are same");
GXF_EXT_FACTORY_END()
