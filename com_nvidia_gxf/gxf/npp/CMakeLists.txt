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
    NAME npp_ext
    SOURCES
        npp.cpp
        nppi_mul_c.cpp
        nppi_set.cpp
    PUBLIC_HEADERS
        nppi_mul_c.hpp
        nppi_set.hpp
    PUBLIC_DEPENDS
        GXF::core
        GXF::std
    PRIVATE_DEPENDS
        CUDA::nppial
        CUDA::nppidei
)
set_target_properties(npp_ext PROPERTIES
    OUTPUT_NAME "gxf_npp"
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
