/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gxf/cuda/tests/test_cuda_helper.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0x469d921446a5431a, 0xb7b59e598d9fc1db, "TestCudaExtension",
                         "Testing Cuda related components", "Nvidia_Gxf", "1.3.0", "NVIDIA");

GXF_EXT_FACTORY_SET_DISPLAY_INFO("Test Cuda Extension", "Cuda", "GXF Cuda Test Extension");

GXF_EXT_FACTORY_ADD(0x638a175249f04a31, 0x821781ce387896db, nvidia::gxf::test::cuda::StreamBasedOps,
                    nvidia::gxf::Codelet, "A base class of cuda streams ops");

GXF_EXT_FACTORY_ADD(0x3a5f0f097bcf49c3, 0xa5437e488fa251f7,
                    nvidia::gxf::test::cuda::StreamTensorGenerator,
                    nvidia::gxf::test::cuda::StreamBasedOps,
                    "Generates cuda tensors with cuda stream id");

GXF_EXT_FACTORY_ADD(0xfbe004939d8d4b4e, 0x9d85e9fddee218be,
                    nvidia::gxf::test::cuda::CublasDotProduct,
                    nvidia::gxf::test::cuda::StreamBasedOps, "Cublas dot product calculation");

GXF_EXT_FACTORY_ADD(0xd877d6f9b72e414b, 0x8e9c52a5aa0bcc1c, nvidia::gxf::test::cuda::HostDotProduct,
                    nvidia::gxf::Codelet, "CPU dot product calculation");

GXF_EXT_FACTORY_ADD(0x4286872912cd4cae, 0xb0ea83d18715d774, nvidia::gxf::test::cuda::MemCpy2Host,
                    nvidia::gxf::test::cuda::StreamBasedOps, "Mem copy device to host async");

GXF_EXT_FACTORY_ADD(0xe578e5bd15434de4, 0x8ed625fd24354ac0, nvidia::gxf::test::cuda::VerifyEqual,
                    nvidia::gxf::Codelet, "Verify whether 2 input tensors same");

GXF_EXT_FACTORY_END()
