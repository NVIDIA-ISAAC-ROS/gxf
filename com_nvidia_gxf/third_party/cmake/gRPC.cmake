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

if(DEFINED gRPC_DIR OR DEFINED gRPC_ROOT)
    find_package(gRPC 1.48.0 CONFIG REQUIRED)
else()
    # TODO builds with its own protobuf, how to converge?

    set(gRPC_INSTALL_DIR "${CMAKE_BINARY_DIR}/gRPC-install")
    set(gRPC_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX:PATH=${gRPC_INSTALL_DIR}")
    list(APPEND gRPC_CMAKE_ARGS "-DgRPC_BUILD_CSHARP_EXT:BOOL=OFF")

    if(CMAKE_BUILD_TYPE)
        set(grpc_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
    else()
        set(grpc_CMAKE_BUILD_TYPE "Release")
    endif()
    list(APPEND gRPC_CMAKE_ARGS "-DCMAKE_BUILD_TYPE:STRING=${grpc_CMAKE_BUILD_TYPE}")
    ExternalProject_Add(
        gRPC
        GIT_REPOSITORY "https://github.com/grpc/grpc"
        GIT_TAG "v1.48.0"
        CMAKE_ARGS ${gRPC_CMAKE_ARGS}
        INSTALL_DIR ${gRPC_INSTALL_DIR}
        GIT_PROGRESS TRUE
        USES_TERMINAL_DOWNLOAD TRUE
        UPDATE_DISCONNECTED TRUE
        DEPENDS GTest
    )

    ExternalProject_Get_Property(gRPC INSTALL_DIR)
    set(gRPC_DIR "${INSTALL_DIR}/lib/cmake/grpc")
    set(absl_DIR "${INSTALL_DIR}/lib/cmake/absl")
    set(Protobuf_DIR "${INSTALL_DIR}/lib/cmake/protobuf")
    unset(INSTALL_DIR)

    list(APPEND GXF_SUPERBUILD_DEPENDS "gRPC")
    list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-Dabsl_DIR:PATH=${absl_DIR}")
    list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-DProtobuf_DIR:PATH=${Protobuf_DIR}")
endif()

list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-DgRPC_DIR:PATH=${gRPC_DIR}")
