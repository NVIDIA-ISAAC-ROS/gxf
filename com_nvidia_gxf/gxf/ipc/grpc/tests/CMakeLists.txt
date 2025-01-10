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
    NAME test_grpc_ext
    SOURCES
        mock_ipc_service_user_ext.cpp
        mock_ipc_service_user.cpp
    PUBLIC_HEADERS
        mock_ipc_service_user.hpp
    PUBLIC_DEPENDS
        GXF::core
        GXF::grpc_ext
        GXF::std
    NO_INSTALL
)

# TODO: https://nvbugspro.nvidia.com/bug/4671357
# 849: 2024-01-24 11:02:42.465 ERROR gxf/gxf/std/yaml_file_loader.cpp@675: Could not create entities from nodes of yaml file : error - yaml-cpp: error at line 19, column 1: bad conversion
gxf_add_gtests(
    EXT_NAME GrpcExt
    SOURCES
        test_grpc.cpp
    DEPENDS
        gRPC::grpc++
        GTest::gtest_main
        GXF::core
        GXF::grpc_ext
        GXF::std
    DATA_FILES
        test_grpc_server.yaml
    BUILD_DEPENDS
        GXF::test_extension
)

# Requires separate GTest executable. Error if combined with previous:
#
# 898: [ RUN      ] TestGrpc.ClientToServerNoYaml
# 898: /workspace/holoscan-sdk/gxf-build-x86_64/GTest-prefix/src/GTest/googletest/src/gtest.cc:2520: Failure
# 898: Failed
# 898: All tests in the same test suite must use the same test fixture
# 898: class, so mixing TEST_F and TEST in the same test suite is
# 898: illegal.  In test suite TestGrpc,
# 898: test ClientToServerNoYaml is defined using TEST_F but
# 898: test ClientToServer is defined using TEST.  You probably
# 898: want to change the TEST to TEST_F or move it to another test
# 898: case.
# 898: [  FAILED  ] TestGrpc.ClientToServerNoYaml (0 ms)
#
# TODO: This test is disabled in Bazel, needs to be fixed or removed.
# 898: [ RUN      ] TestGrpc.ClientToServerNoYaml
# 898: 2024-08-26 02:26:10.324 ERROR /workspace/holoscan-sdk/gxf/gxf/core/parameter_storage.hpp@121: Attempting to set invalid parameter type for [port] with type [int]
# 898: 2024-08-26 02:26:10.324 PANIC /workspace/holoscan-sdk/gxf/gxf/ipc/grpc/tests/test_grpc_no_yaml.cpp@75: GXF operation failed: GXF_PARAMETER_INVALID_TYPE
#
# gxf_add_gtests(
#     EXT_NAME GrpcExtNoYAML
#     SOURCES
#         test_grpc_no_yaml.cpp
#     DEPENDS
#         gRPC::grpc++
#         GTest::gtest_main
#         GXF::core
#         GXF::grpc_ext
#         GXF::std
#     DATA_FILES
#         test_grpc_server.yaml
#     BUILD_DEPENDS
#         GXF::test_extension
# )

gxf_add_gxe_tests(
    GROUP_NAME GrpcExtServerClient
    APP_FILES test_grpc_server_client.yaml
    EXT_DEPENDS
        GXF::std
        GXF::grpc_ext
        GXF::test_grpc_ext
)
