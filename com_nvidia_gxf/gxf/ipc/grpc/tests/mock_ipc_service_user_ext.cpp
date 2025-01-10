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

#include "gxf/ipc/grpc/tests/mock_ipc_service_user.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()
  GXF_EXT_FACTORY_SET_INFO(0x24bbf0644e67429f, 0xb32de395e00f6a70, "TestIPCServerClientExtension",
                           "Helper extension with components used to test IPC server and client",
                           "NVIDIA", "0.0.1", "LICENSE");
  GXF_EXT_FACTORY_ADD(0xbd493e1c3edf4e36, 0xa3e17f81082d9150,
                       nvidia::gxf::MockIPCServiceUser, nvidia::gxf::Component,
                       "An example IPCServer owner or implementer");
  GXF_EXT_FACTORY_ADD(0xe97040f3075e4512, 0xa3c7fbbcf159c081,
                      nvidia::gxf::MockIPCClientUser, nvidia::gxf::Codelet,
                      "An example IPCClient user");
GXF_EXT_FACTORY_END()
