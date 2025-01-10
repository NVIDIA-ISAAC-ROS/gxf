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

if(rmm_DIR OR rmm_ROOT)
    find_package(rmm REQUIRED)
else()
    set(RMM_INSTALL_PATH "${CMAKE_BINARY_DIR}/rmm-install")
    set(rmm_PATCH_FILEPATH "${CMAKE_SOURCE_DIR}/third_party/rmm.patch")
    ExternalProject_Add(
        rmm
        URL "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/rmm/v24.04.00.tar.gz"
        URL_HASH "SHA256=bb20877c8d92b322dbcb348c2009040784189d3d3c48f93011e13c1b34f6a22f"
        PATCH_COMMAND git apply -p0 "${rmm_PATCH_FILEPATH}"
        INSTALL_DIR ${RMM_INSTALL_PATH}
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX:PATH=${RMM_INSTALL_PATH}
            -DBUILD_TESTS:BOOL=OFF
    )
    # fmt and spdlog built and installed with rmm
    add_custom_target(fmt DEPENDS rmm)
    add_custom_target(spdlog DEPENDS rmm)

    ExternalProject_Get_Property(rmm INSTALL_DIR)
    set(rmm_DIR "${INSTALL_DIR}/lib/cmake/rmm")
    set(fmt_DIR "${INSTALL_DIR}/lib/cmake/fmt")
    set(spdlog_DIR "${INSTALL_DIR}/lib/cmake/spdlog")
    unset(INSTALL_DIR)

    list(APPEND GXF_SUPERBUILD_DEPENDS "rmm")
endif()

list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-Drmm_DIR:PATH=${rmm_DIR}")
list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-Dfmt_DIR:PATH=${fmt_DIR}")
list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-Dspdlog_DIR:PATH=${spdlog_DIR}")
