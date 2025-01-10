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

# Configures the build tree to run an "outer" build of project dependencies
# first, then initiate an "inner" project build of GXF.
# For more information on the CMake "superbuild" structure, refer to:
# https://www.kitware.com/cmake-superbuilds-git-submodules/ .
# Called by the root `CMakeLists.txt`.

unset(GXF_SUPERBUILD_DEFINED_ARGS)
unset(GXF_SUPERBUILD_DEPENDS)

# Cache and reset values to avoid passing to dependencies
set(GXF_INNER_BUILD_TESTING ${BUILD_TESTING})
set(BUILD_TESTING OFF)

find_package(Threads REQUIRED)
find_package(CUDAToolkit
    COMPONENTS nvtx3
    REQUIRED
)
if(CMAKE_CUDA_COMPILER)
    list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-DCMAKE_CUDA_COMPILER:FILEPATH=${CMAKE_CUDA_COMPILER}")
endif()

include(FetchContent)
include(ExternalProject)
include(third_party/cmake/yaml-cpp.cmake)
include(third_party/cmake/nlohmann_json.cmake)
include(third_party/cmake/magic_enum.cmake)
include(third_party/cmake/dlpack.cmake)
include(third_party/cmake/nvsci.cmake)
include(third_party/cmake/ucx.cmake)
include(third_party/cmake/Boost.cmake)
include(third_party/cmake/cpprestsdk.cmake)
include(third_party/cmake/pybind11.cmake)

include(third_party/cmake/lss.cmake)
include(third_party/cmake/breakpad.cmake)
include(third_party/cmake/gflags.cmake)
include(third_party/cmake/GTest.cmake)

include(third_party/cmake/gRPC.cmake)
#include(third_party/cmake/Protobuf.cmake) # Protobuf builds with gRPC

include(third_party/cmake/rmm.cmake)

add_custom_target(gxf_dependencies
    COMMAND cmake -E touch gxf_dependencies.stamp
    COMMENT "Building GXF outer build of dependencies without GXF sources"
    DEPENDS ${GXF_SUPERBUILD_DEPENDS}
)

if(GXF_INNER_BUILD)
    list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-DGXF_SUPERBUILD:BOOL=OFF")
    list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-DGXF_INNER_BUILD:BOOL=ON")
    list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-DBUILD_TESTING:BOOL=${GXF_INNER_BUILD_TESTING}")
    list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-DCMAKE_INSTALL_PREFIX:STRING=${CMAKE_BINARY_DIR}/gxf-install")
    if(BUILD_SHARED_LIBS)
        list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}")
    endif()
    if(CMAKE_TOOLCHAIN_FILE)
        list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}")
    endif()
    if(CMAKE_BUILD_TYPE)
        list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
    endif()
    if(GXF_PRESET_NAME)
        list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-DGXF_PRESET_NAME=${GXF_PRESET_NAME}")
    endif()
    message(STATUS "GXF_SUPERBUILD_ARGS ${GXF_SUPERBUILD_DEFINED_ARGS}")
    ExternalProject_Add(
        gxf
        DEPENDS ${GXF_SUPERBUILD_DEPENDS}
        SOURCE_DIR "${CMAKE_SOURCE_DIR}"
        BINARY_DIR "${CMAKE_BINARY_DIR}/gxf-build"
        CMAKE_ARGS ${GXF_SUPERBUILD_DEFINED_ARGS}
        USES_TERMINAL_CONFIGURE TRUE
        USES_TERMINAL_BUILD TRUE
    )
endif()
