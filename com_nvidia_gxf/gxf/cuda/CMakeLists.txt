# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

set(GXF_CUDA_SOURCES
    cuda.cpp
    cuda_allocator.cpp
    cuda_event.cpp
    cuda_scheduling_terms.cpp
    cuda_stream.cpp
    cuda_stream_pool.cpp
    cuda_stream_sync.cpp
    stream_ordered_allocator.cpp)
set(GXF_CUDA_HEADERS
    cuda_allocator.hpp
    cuda_buffer.hpp
    cuda_common.hpp
    cuda_event.hpp
    cuda_scheduling_terms.hpp
    cuda_stream.hpp
    cuda_stream_id.hpp
    cuda_stream_pool.hpp
    cuda_stream_sync.hpp
    stream_ordered_allocator.hpp)
add_library(cuda SHARED ${GXF_CUDA_SOURCES})
set_target_properties(cuda
  PROPERTIES
    OUTPUT_NAME "gxf_cuda"
    PUBLIC_HEADER "${GXF_CUDA_HEADERS}"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../core:$ORIGIN/../std"
)

target_compile_features(cuda PUBLIC cuda_std_20)
target_compile_options(cuda PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_include_directories(cuda PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
                                       $<INSTALL_INTERFACE:include>)
target_link_libraries(cuda PUBLIC GXF::common GXF::core GXF::std CUDA::cudart)
add_library(GXF::cuda ALIAS cuda)

include(GNUInstallDirs)
install(
  TARGETS cuda
  EXPORT gxfCoreTargets
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gxf/cuda
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/cuda
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/cuda
  COMPONENT cuda)
set(GXF_EXTENSION_LIBRARY_TARGETS "cuda;${GXF_EXTENSION_LIBRARY_TARGETS}" PARENT_SCOPE)

#######################################################################################
# Pybind11 extensions
#######################################################################################

pybind11_add_module(cuda_pybind SHARED WITHOUT_SOABI "bindings/cuda.cpp")
set_target_properties(cuda_pybind PROPERTIES OUTPUT_NAME "cuda_pybind" PREFIX "")
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
