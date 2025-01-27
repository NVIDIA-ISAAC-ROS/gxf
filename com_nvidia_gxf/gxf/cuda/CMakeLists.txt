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
    NAME cuda
    SOURCES
        cuda.cpp
        cuda_allocator.cpp
        cuda_event.cpp
        cuda_scheduling_terms.cpp
        cuda_stream.cpp
        cuda_stream_pool.cpp
        cuda_stream_sync.cpp
        stream_ordered_allocator.cpp
    PUBLIC_HEADERS
        cuda_allocator.hpp
        cuda_buffer.hpp
        cuda_common.hpp
        cuda_event.hpp
        cuda_scheduling_terms.hpp
        cuda_stream.hpp
        cuda_stream_id.hpp
        cuda_stream_pool.hpp
        cuda_stream_sync.hpp
        stream_ordered_allocator.hpp
    PUBLIC_DEPENDS
        GXF::common
        GXF::core
        GXF::std
        CUDA::cudart
)

#######################################################################################
# Pybind11 extensions
#######################################################################################

pybind11_add_module(cuda_pybind SHARED "bindings/cuda.cpp")
target_link_libraries(cuda_pybind PUBLIC GXF::common GXF::core GXF::std CUDA::cudart)


install(
    TARGETS cuda_pybind
    EXPORT gxfCoreTargets
    PUBLIC_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gxf/cuda
    ARCHIVE
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/cuda
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/cuda
    COMPONENT Core
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
