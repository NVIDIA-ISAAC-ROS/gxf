# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

gxf_core_add_extension_library(
    NAME test_http
    SOURCES
        mock_http_service.cpp
        test_http.cpp
    PUBLIC_HEADERS
        mock_http_service.hpp
    PUBLIC_DEPENDS
        GXF::core
        GXF::http
        GXF::std
    NO_INSTALL
)

gxf_add_gtests(
    EXT_NAME HTTP
    SOURCES
        test_http_no_yaml.cpp
    DEPENDS
        GTest::gtest_main
        GXF::core
        GXF::http
)

gxf_add_gxe_tests(
    GROUP_NAME HTTPServer
    APP_FILES
        test_http_server_client.yaml
        test_http_server_ipc_client.yaml
    EXT_DEPENDS
        GXF::std
        GXF::http
        GXF::test_http
)
