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
    NAME clock_sync_helpers
    SOURCES
        test_clock_sync_helpers.cpp
        test_clock_sync_helpers.hpp
        test_clock_sync_helpers_factory.cpp
    PUBLIC_DEPENDS GXF::std
    NO_INSTALL
)
set_target_properties(clock_sync_helpers
    PROPERTIES OUTPUT_NAME "gxf_test_clock_sync_helpers_factory"
)

gxf_add_gtests(
    EXT_NAME Network
    SOURCES
        test_clock_sync.cpp
        test_clock_sync_ebs.cpp
        test_tcp.cpp
        test_tcp_ebs.cpp
    DATA_FILES
        test_clock_sync_primary_ebs.yaml
        test_clock_sync_primary.yaml
        test_clock_sync_secondary_ebs.yaml
        test_clock_sync_secondary.yaml
        test_tcp_client_bidirectional_ebs.yaml
        test_tcp_client_bidirectional_stress.yaml
        test_tcp_client_bidirectional.yaml
        test_tcp_client_connected_interrupt.yaml
        test_tcp_client_reconnection_server_ebs.yaml
        test_tcp_client_reconnection_server.yaml
        test_tcp_client_sink_async_ebs.yaml
        test_tcp_client_sink_async.yaml
        test_tcp_client_sink_ebs.yaml
        test_tcp_client_sink.yaml
        test_tcp_client_source_ebs.yaml
        test_tcp_client_source.yaml
        test_tcp_client_stress.yaml
        test_tcp_client_unconnected_interrupt_ebs.yaml
        test_tcp_client_unconnected_interrupt.yaml
        test_tcp_server_bidirectional_ebs.yaml
        test_tcp_server_bidirectional_stress.yaml
        test_tcp_server_bidirectional.yaml
        test_tcp_server_connected_interrupt.yaml
        test_tcp_server_receiver_push_client_ebs.yaml
        test_tcp_server_receiver_push_client.yaml
        test_tcp_server_receiver_push_server_ebs.yaml
        test_tcp_server_receiver_push_server.yaml
        test_tcp_server_reconnection_client_ebs.yaml
        test_tcp_server_reconnection_client.yaml
        test_tcp_server_sink_ebs.yaml
        test_tcp_server_sink.yaml
        test_tcp_server_source_async_ebs.yaml
        test_tcp_server_source_async.yaml
        test_tcp_server_source_ebs.yaml
        test_tcp_server_source.yaml
        test_tcp_server_stress.yaml
        test_tcp_server_unconnected_interrupt_ebs.yaml
        test_tcp_server_unconnected_interrupt.yaml
    DEPENDS
        GXF::core
        GXF::network
        GXF::sample
        GXF::serialization
        GXF::std
        GTest::gtest_main
    BUILD_DEPENDS
        clock_sync_helpers
)
