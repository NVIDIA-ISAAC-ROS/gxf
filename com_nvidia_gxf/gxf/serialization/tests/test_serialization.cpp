/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gxf/serialization/tests/serialization_tester.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()
  GXF_EXT_FACTORY_SET_INFO(0xc8dfd61abe545fd4, 0xf828df87e6b0162b, "TestSerializationExtension",
                           "Extension for testing serialization extension",
                           "NVIDIA", "1.2.0", "NVIDIA");
  GXF_EXT_FACTORY_SET_DISPLAY_INFO("Test Serialization Extension", "Serialization",
                                 "GXF Serialization Extension");
  GXF_EXT_FACTORY_ADD(0xdb9f4ca3f121471b, 0xb0a0d49b89a0dc19,
                      nvidia::gxf::test::SerializationTester, nvidia::gxf::Codelet,
                      "Codelet used to test component serializers");
GXF_EXT_FACTORY_END()
