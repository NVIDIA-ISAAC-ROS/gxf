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
    NAME serialization_test_extension
    SOURCES
        test_serialization.cpp
        serialization_tester.cpp
    PUBLIC_HEADERS
        serialization_tester.hpp
    PUBLIC_DEPENDS
        GXF::common
        GXF::core
        GXF::serialization
        GXF::std
)

gxf_add_gtests(
    EXT_NAME Serialization
    SOURCES
        test_entity_record_replay.cpp
        test_file.cpp
    DATA_FILES
        test_entity_recorder.yaml
        test_entity_replayer.yaml
        test_std_component_serializer_device.yaml
        test_std_component_serializer_host.yaml
        test_std_component_serializer_system.yaml
    DEPENDS
        GTest::gtest_main
        GXF::serialization
    BUILD_DEPENDS
        serialization_test_extension
)
