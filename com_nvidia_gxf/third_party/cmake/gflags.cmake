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

if(gflags_DIR OR gflags_ROOT)
    find_package(gflags REQUIRED)
else()
    set(gflags_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/gflags-prefix")
    set(gflags_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/gflags-install")
    set(gflags_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX:PATH=${gflags_INSTALL_PREFIX}")
    list(APPEND gflags_CMAKE_ARGS "-DCMAKE_POSITION_INDEPENDENT_CODE:PATH=${CMAKE_POSITION_INDEPENDENT_CODE}")
    list(APPEND gflags_CMAKE_ARGS "-DGFLAGS_REGISTER_INSTALL_PREFIX:BOOL=OFF")

    ExternalProject_Add(
        gflags
        URL "https://developer.nvidia.com/isaac/download/third_party/gflags-e292e0452fcfd5a8ae055b59052fc041cbab4abf-tar-gz"
        URL_HASH "SHA256=a4c5171355e67268b4fd2f31c3f7f2d125683d12e0686fc14893a3ca8c803659"
        DOWNLOAD_DIR ${gflags_DOWNLOAD_DIR}
        CMAKE_ARGS ${gflags_CMAKE_ARGS}
        INSTALL_DIR ${gflags_INSTALL_PREFIX}
    )
    ExternalProject_Get_Property(gflags INSTALL_DIR)
    set(gflags_DIR "${INSTALL_DIR}/lib/cmake/gflags")
    unset(INSTALL_DIR)
    list(APPEND GXF_SUPERBUILD_DEPENDS "gflags")
endif()

list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-Dgflags_DIR:PATH=${gflags_DIR}")
