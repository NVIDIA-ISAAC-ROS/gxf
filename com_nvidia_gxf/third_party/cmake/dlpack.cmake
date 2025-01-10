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

if(dlpack_DIR OR dlpack_ROOT)
    find_package(dlpack REQUIRED)
else()
    set(dlpack_CMAKE_ARGS "-DBUILD_MOCK:BOOL=OFF")
    set(dlpack_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/dlpack-install")
    list(APPEND dlpack_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX:PATH=${dlpack_INSTALL_PREFIX}")

    ExternalProject_Add(
        dlpack
        URL "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/dlpack/v0.8.tar.gz"
        URL_HASH "SHA256=cf965c26a5430ba4cc53d61963f288edddcd77443aa4c85ce722aaf1e2f29513"
        CMAKE_ARGS ${dlpack_CMAKE_ARGS}
        INSTALL_DIR ${dlpack_INSTALL_PREFIX}
    )
    ExternalProject_Get_Property(dlpack INSTALL_DIR)
    set(dlpack_DIR "${INSTALL_DIR}/lib/cmake/dlpack")
    unset(INSTALL_DIR)
    list(APPEND GXF_SUPERBUILD_DEPENDS "dlpack")
endif()

list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-Ddlpack_DIR:PATH=${dlpack_DIR}")

