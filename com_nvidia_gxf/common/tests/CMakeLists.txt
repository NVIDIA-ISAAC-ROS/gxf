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
    EXT_NAME Common
    SOURCES
        test_endian.cpp
        test_expected.cpp
        test_fixed_map.cpp
        test_fixed_string.cpp
        test_fixed_vector.cpp
        test_iterator.cpp
        test_logger.cpp
        test_memory_utils.cpp
        test_span.cpp
        test_strong_type.cpp
        test_type_name.cpp
        test_type_utils.cpp
        test_unique_index_map.cpp
    DEPENDS
        GTest::gtest_main
        GXF::common
        GXF::core
        GXF::logger
)
