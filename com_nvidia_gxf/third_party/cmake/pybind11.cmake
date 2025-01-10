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

if(DEFINED pybind11_DIR OR DEFINED pybind11_ROOT)
    find_package(pybind11 REQUIRED)
else()

    set(pybind11_INSTALL_DIR "${CMAKE_BINARY_DIR}/pybind11-install")
    set(pybind11_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX:PATH=${pybind11_INSTALL_DIR}")
    list(APPEND pybind11_CMAKE_ARGS "-DPYBIND11_TEST:BOOL=OFF")

    ExternalProject_Add(
        pybind11
        URL "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/external/pybind11-2.11.1.tar.gz"
        URL_HASH "SHA256=4744701624538da603dde2b533c5a56fac778ea4773650332fe6701b25f191aa"
        CMAKE_ARGS "${pybind11_CMAKE_ARGS}"
        INSTALL_DIR "${pybind11_INSTALL_DIR}"
    )

    ExternalProject_Get_Property(pybind11 INSTALL_DIR)
    set(pybind11_DIR "${INSTALL_DIR}/share/cmake/pybind11")
    unset(INSTALL_DIR)

    list(APPEND GXF_SUPERBUILD_DEPENDS "pybind11")
endif()

list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-Dpybind11_DIR:PATH=${pybind11_DIR}")
