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

include(GoogleTest)

# Separate executables to avoid collisions among `MockLogger`

add_executable(GXFCommonLoggerTest test_logger.cpp)
target_include_directories(GXFCommonLoggerTest PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(GXFCommonLoggerTest
    GTest::gtest_main
    GXF::logger
)

add_executable(GXFCommonGXFLoggerTest test_gxf_logger.cpp)
target_include_directories(GXFCommonGXFLoggerTest PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(GXFCommonGXFLoggerTest
    GTest::gtest_main
    GXF::core
    GXF::logger
)

gtest_add_tests(GXFCommonLoggerTest
    SOURCES test_logger.cpp
)
gtest_add_tests(GXFCommonGXFLoggerTest
    SOURCES test_gxf_logger.cpp
)
