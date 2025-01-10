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

cmake_minimum_required(VERSION 3.24.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(GXF_PRESET_NAME)
    message(STATUS "Configuring GXF for ${GXF_PRESET_NAME}")
else()
    message(FATAL_ERROR "No GXF configuration found. Please specify a valid preset with `cmake --preset <name>`.")
endif()

#####################################################
# Options
#####################################################
option(BUILD_TESTING OFF)
option(BUILD_SHARED_LIBS ON)

set(THREADS_PREFER_PTHREAD_FLAG ON)
# Required for GXF extension shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#####################################################
# Dependency superbuild infrastructure
#####################################################

include(CMakeDependentOption)
option(GXF_SUPERBUILD "Perform an outer build of GXF dependencies." ON)
cmake_dependent_option(GXF_INNER_BUILD "Perform the GXF inner build." ON "GXF_SUPERBUILD" OFF)

if(GXF_SUPERBUILD)
    message(STATUS "Configuring outer build of GXF dependencies.")
    project(gxf_superbuild
        DESCRIPTION
            "Builds the NVIDIA Graphical eXecution Framework (GXF) and its dependencies."
        )
    include(cmake/Superbuild.cmake)
    return()
endif()

#####################################################
# Inner build
#####################################################

message(STATUS "Configuring inner build of GXF libraries.")
project(
    gxf
    VERSION 4.1.0
    DESCRIPTION
        "The NVIDIA Graphical eXecution Framework (GXF) is a framework supporting component-based programming for streaming data processing pipelines."
    HOMEPAGE_URL "https://docs.nvidia.com/holoscan/sdk-user-guide/gxf/doc/index.html"
    LANGUAGES C CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")
find_package(Threads REQUIRED)
set(CUDA_COMPONENTS_REQUIRED nvtx3 cudart npp)
if(BUILD_TESTING)
    list(APPEND CUDA_COMPONENTS_REQUIRED cublas)
endif()


find_package(CUDAToolkit 12.2
    COMPONENTS ${CUDA_COMPONENTS_REQUIRED}
    REQUIRED
)
# Backport to support `nvtx3` target in CMake<=3.24.
# CMake<=3.24 "FindCUDAToolkit" does not initialize CUDA::nvtx3,
# so we must add the target ourselves.
# TODO: Remove this workaround and enforce CMake>=3.25.
if(NOT TARGET CUDA::nvtx3)
    find_file(
        CUDA_nvtx3_INCLUDE_DIR
        NAME "nvtx3"
        HINTS ${CUDAToolkit_INCLUDE_DIRS}
        REQUIRED
    )
    add_library(CUDA::nvtx3 INTERFACE IMPORTED)
    set_target_properties(CUDA::nvtx3
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CUDA_nvtx3_INCLUDE_DIR}
    )
endif()

find_package(yaml-cpp REQUIRED)
add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
find_package(nlohmann_json REQUIRED)
find_package(breakpad REQUIRED)
find_package(gflags REQUIRED)
add_library(gflags::gflags_static ALIAS gflags_static)
find_package(magic_enum REQUIRED)
find_package(dlpack REQUIRED)
find_package(nvsci REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(ucx CONFIG REQUIRED)
find_package(cpprestsdk 2.10.18 REQUIRED)

find_package(rmm 24.04 REQUIRED)
 # RMM 24.04.0 pollutes the reserved PACKAGE_VERSION variable.
 # https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html
unset(PACKAGE_VERSION)

find_package(Python3
    COMPONENTS Interpreter Development
    REQUIRED
)
find_package(pybind11 REQUIRED)

find_package(GTest REQUIRED)
find_package(gRPC CONFIG REQUIRED)

if(BUILD_TESTING OR GXF_BUILD_TESTING)
    enable_testing()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(UseGXF)
include(UseGXFInternal)
install(
    FILES cmake/UseGXF.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GXF
)

add_subdirectory(common)
add_subdirectory(gxf)

# TODO: JIRA 3105
# Build remaining directories with CMake
# add_subdirectory(sample)
# add_subdirectory(packaging)
# ...

include(cmake/InstallAndExport.cmake)
