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

if(yaml-cpp_DIR OR yaml-cpp_ROOT)
    find_package(yaml-cpp REQUIRED)
else()
    set(YAML_BUILD_ARGS "-DYAML_CPP_BUILD_TESTS:BOOL=OFF")
    list(APPEND YAML_BUILD_ARGS "-DYAML_CPP_BUILD_TOOLS:BOOL=ON")
    list(APPEND YAML_BUILD_ARGS "-DYAML_CPP_BUILD_CONTRIB:BOOL=OFF")
    list(APPEND YAML_BUILD_ARGS "-DYAML_BUILD_SHARED_LIBS:BOOL=OFF")
    list(APPEND YAML_BUILD_ARGS "-DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_BINARY_DIR}/yaml-cpp-install")
    list(APPEND YAML_BUILD_ARGS "-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=${CMAKE_POSITION_INDEPENDENT_CODE}")

    ExternalProject_Add(
        yaml-cpp
        URL https://developer.nvidia.com/isaac/download/third_party/yaml-cpp-0-6-3-tar-gz
        URL_HASH "SHA256=f38a7a7637993943c4c890e352b1fa3f3bf420535634e9a506d9a21c3890d505"
        CMAKE_ARGS ${YAML_BUILD_ARGS}
    )
    ExternalProject_Get_Property(yaml-cpp SOURCE_DIR)
    set(yaml-cpp_DIR "${CMAKE_BINARY_DIR}/yaml-cpp-install/lib/cmake/yaml-cpp")
    unset(SOURCE_DIR)

    list(APPEND GXF_SUPERBUILD_DEPENDS "yaml-cpp")
endif()

list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-Dyaml-cpp_DIR:PATH=${yaml-cpp_DIR}")
