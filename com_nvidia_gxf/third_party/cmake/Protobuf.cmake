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

# Google Protobuf
# TODO: should we ever actually build this ourselves?
# We prefer installation via apt-get

if(Protobuf_DIR OR Protobuf_ROOT)
    find_package(Protobuf CONFIG REQUIRED)

    # Create custom target for downstream external projects to depend on
    add_custom_target(Protobuf
        COMMAND cmake -E touch Protobuf.stamp
        COMMENT "Found system Protobuf dependency"
    )
else()
    set(Protobuf_CMAKE_ARGS "-Dprotobuf_BUILD_TESTS:BOOL=OFF")
    set(Protobuf_INSTALL_DIR "${CMAKE_BINARY_DIR}/Protobuf-install")
    list(APPEND Protobuf_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX:PATH=${Protobuf_INSTALL_DIR}")
    list(APPEND Protobuf_CMAKE_ARGS "-DCMAKE_POSITION_INDEPENDENT_CODE:PATH=${CMAKE_POSITION_INDEPENDENT_CODE}")

    ExternalProject_Add(
        Protobuf
        GIT_REPOSITORY https://github.com/protocolbuffers/protobuf
        GIT_TAG "v21.7"
        CMAKE_ARGS ${Protobuf_CMAKE_ARGS}
        INSTALL_DIR ${Protobuf_INSTALL_DIR}
    )
    set(Protobuf_DIR "${Protobuf_INSTALL_DIR}/lib/cmake/protobuf")
    list(APPEND GXF_SUPERBUILD_DEPENDS "Protobuf")
endif()

list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-DProtobuf_DIR:PATH=${Protobuf_DIR}")
