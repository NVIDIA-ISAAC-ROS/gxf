/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gxf/ipc/http/tests/mock_http_service.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()
  GXF_EXT_FACTORY_SET_INFO(0xd8f8ae7a8a824500, 0xb380dabd19c7053c, "TestHttpHelperExtension",
                           "Helper extension with components used to test http components",
                           "NVIDIA", "0.2.0", "LICENSE");
  GXF_EXT_FACTORY_ADD(0xaf78593bf4e14d58, 0xae38d91710e7ebaf,
                       nvidia::gxf::MockHttpService, nvidia::gxf::Component,
                       "An example http service owner object");
  GXF_EXT_FACTORY_ADD(0x70a44b40759c4d12, 0x8e598b01854fd96b,
                      nvidia::gxf::MockHttpClient, nvidia::gxf::Codelet,
                      "An example http client user object");
  GXF_EXT_FACTORY_ADD(0xe2be0e4abcaf4f87, 0xaff3ee2afaea07fc,
                       nvidia::gxf::MockHttpIPCClient, nvidia::gxf::Codelet,
                       "An example http client user object");
GXF_EXT_FACTORY_END()
