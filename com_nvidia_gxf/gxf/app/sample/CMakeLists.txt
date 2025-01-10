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

gxf_add_gtests(
    EXT_NAME AppSample
    SOURCES
        ping_graph_multi_broadcast.cpp
        ping_graph_multi.cpp
        ping_graph_multi_gather.cpp
        ping_graph_runtime.cpp
        ping_graph_simple.cpp
        ping_multi_segment_example.cpp
        ping_segment_example.cpp
        tensor_copier_device_to_host.cpp
        tensor_copier_device_to_system.cpp
        tensor_copier_host_to_device.cpp
        tensor_copier_host_to_system.cpp
        tensor_copier_system_to_device.cpp
        tensor_copier_system_to_host.cpp
        test_factorial.cpp
        test_hello_world.cpp
        test_hello_world_multi.cpp
        test_synchronization.cpp
    DEPENDS
        GTest::gtest_main
        GXF::app
        GXF::common
        GXF::core
        GXF::sample
        GXF::std
        GXF::test_extension
    DATA_FILES
        tensor_copier_parameters.yaml
)
