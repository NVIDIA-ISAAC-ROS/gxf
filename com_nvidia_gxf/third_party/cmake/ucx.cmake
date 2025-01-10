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

if(ucx_DIR OR ucx_ROOT)
    find_package(ucx REQUIRED)
else()
    if(NOT ucx_URL)
        message(FATAL_ERROR "Selected preset does not provide UCX URL source")
    endif()
    set(ucx_TARGETS_CMAKE_FILEPATH "${ucx_INSTALL_PREFIX}lib/cmake/ucx/ucx-targets.cmake")
    configure_file(
        "${CMAKE_SOURCE_DIR}/third_party/cmake/ucx.patch.in"
        "${CMAKE_BINARY_DIR}/third_party/cmake/ucx.patch"
        @ONLY
    )

    ExternalProject_Add(
        ucx
        URL ${ucx_URL}
        URL_HASH "SHA256=${ucx_HASH}"
        PATCH_COMMAND patch -p0 < ${CMAKE_BINARY_DIR}/third_party/cmake/ucx.patch
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
    )

    set(ucx_DIR "${CMAKE_BINARY_DIR}/ucx-prefix/src/ucx/${ucx_INSTALL_PREFIX}/lib/cmake/ucx")
    list(APPEND GXF_SUPERBUILD_DEPENDS "ucx")
endif()

list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-Ducx_DIR:PATH=${ucx_DIR}")
