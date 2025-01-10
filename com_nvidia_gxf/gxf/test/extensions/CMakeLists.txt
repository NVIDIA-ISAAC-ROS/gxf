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
    NAME test_extension
    SOURCES
        test.cpp
    PUBLIC_HEADERS
        test_helpers.hpp
        test_metric.hpp
        test_parameters.hpp
    PUBLIC_DEPENDS
        GTest::gtest_main
        GXF::common
        GXF::core
        GXF::std
    PRIVATE_DEPENDS
        GXF::test::components
)
