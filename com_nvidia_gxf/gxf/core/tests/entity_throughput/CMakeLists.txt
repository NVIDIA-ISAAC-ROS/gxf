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
    EXT_NAME CoreEntityThroughput
    SOURCES
        test_create_entities.cpp
        test_create_entities_int.cpp
        test_create_entities_int_ten.cpp
        test_create_entity_rate.cpp
    DEPENDS
        GTest::gtest_main
        GXF::common
        GXF::core
        GXF::std
        GXF::test_load_extension
    BUILD_DEPENDS
        GXF::test_extension
        gxf_core_manifest
)
