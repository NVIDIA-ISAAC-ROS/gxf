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
    NAME test_stream_sync_cuda
    SOURCES
        test_stream_sync_cuda_ext.cpp
    PUBLIC_HEADERS
        test_gxf_stream_sync_cuda_helper.hpp
    PUBLIC_DEPENDS
        CUDA::cublas
        GXF::common
        GXF::cuda
        GXF::std
        GXF::stream
)

gxf_add_gtests(
    EXT_NAME Stream
    SOURCES
        test_gxf_stream_unit.cpp
    DEPENDS
        GXF::core
        GXF::stream
        GTest::gtest_main
)

gxf_add_gxe_tests(
    GROUP_NAME StreamCUDADotProduct
    APP_FILES test_gxf_stream_cuda_dotproduct.yaml
    EXT_DEPENDS
        GXF::std
        GXF::cuda
        GXF::stream
        GXF::test_extension
        test_stream_sync_cuda
)
