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

if(nlohmann_json_DIR OR nlohmann_json_ROOT)
    find_package(nlohmann_json REQUIRED)
else()
    # TODO: The Bazel artifactory package for `nlohmann-json-3-10-5` appears to exclude
    # necessary CMake files.
    # As a temporary workaround, we pull sources from GitHub instead.
    set(nlohmann_CMAKE_ARGS "-DJSON_BuildTests:BOOL=OFF")
    list(APPEND nlohmann_CMAKE_ARGS "-DJSON_CI:BOOL=OFF")
    list(APPEND nlohmann_CMAKE_ARGS "-DJSON_Install:BOOL=ON")
    set(nlohmann_json_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/nlohmann_json-install")
    list(APPEND nlohmann_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX:PATH=${nlohmann_json_INSTALL_PREFIX}")

    ExternalProject_Add(
        nlohmann_json
        GIT_REPOSITORY "https://github.com/nlohmann/json"
        GIT_TAG "v3.10.5"
        CMAKE_ARGS ${nlohmann_CMAKE_ARGS}
        INSTALL_DIR ${nlohmann_json_INSTALL_PREFIX}
    )
    ExternalProject_Get_Property(nlohmann_json INSTALL_DIR)
    set(nlohmann_json_DIR "${INSTALL_DIR}/lib/cmake/nlohmann_json")
    unset(INSTALL_DIR)

    list(APPEND GXF_SUPERBUILD_DEPENDS "nlohmann_json")
endif()

list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-Dnlohmann_json_DIR:PATH=${nlohmann_json_DIR}")
