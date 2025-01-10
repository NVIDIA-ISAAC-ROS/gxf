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
    EXT_NAME rmm
    SOURCES test_memory_pool.cpp
    DEPENDS
        GXF::rmm
        GXF::std
        GXF::test_extension
        GTest::gtest_main
)

gxf_add_gxe_tests(
    GROUP_NAME rmm_yaml_test
    APP_FILES
        test_rmm_allocator.yaml
        test_rmm_allocator_stress_test.yaml
        test_cuda_stream_dotproduct_rmm_allocator.yaml
    EXT_DEPENDS
        std
        sample
        cuda
        test_extension
        test_cuda
        rmm
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_stress_cuda_stream_dotproduct_rmm_allocator_gen.yaml
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_stress_cuda_stream_dotproduct_rmm_allocator.py
        ${CMAKE_CURRENT_BINARY_DIR}/test_stress_cuda_stream_dotproduct_rmm_allocator_gen.yaml
)

# TODO add CTest entry for stress test
