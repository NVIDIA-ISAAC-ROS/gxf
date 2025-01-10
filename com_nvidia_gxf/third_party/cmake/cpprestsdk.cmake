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

if(DEFINED cpprestsdk_DIR OR DEFINED cpprestsdk_ROOT)
    find_package(cpprestsdk REQUIRED)
else()
    # FIXME: Default BoringSSL (b9232f9e) does not install or generate CMake config
    # so we are using (2fc6d383) for CMake build
    # Requires Go:
    # https://go.dev/wiki/Ubuntu
    # sudo add-apt-repository ppa:longsleep/golang-backports
    # sudo apt update
    # sudo apt install golang-go
    # OR
    # sudo snap install --classic go
    set(BoringSSL_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/BoringSSL-install")
    set(BoringSSL_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX:PATH=${BoringSSL_INSTALL_PREFIX}")
    list(APPEND BoringSSL_CMAKE_ARGS "-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON")
    ExternalProject_Add(
        BoringSSL
        GIT_REPOSITORY "https://github.com/google/boringssl"
        GIT_TAG "2fc6d38391cb76839c76b2a462619e7d69fd998d"
        CMAKE_ARGS ${BoringSSL_CMAKE_ARGS}
        INSTALL_DIR ${BoringSSL_INSTALL_PREFIX}
        GIT_PROGRESS TRUE
        USES_TERMINAL_DOWNLOAD TRUE
        UPDATE_DISCONNECTED TRUE
    )
    list(APPEND cpprestsdk_CMAKE_ARGS
        "-DOpenSSL_DIR:PATH=${BoringSSL_INSTALL_PREFIX}/lib/cmake/OpenSSL"
    )
    list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-DOpenSSL_DIR:PATH=${BoringSSL_INSTALL_PREFIX}/lib/cmake/OpenSSL")

    set(cpprestsdk_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/cpprestsdk-install")
    list(APPEND cpprestsdk_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX:PATH=${cpprestsdk_INSTALL_PREFIX}")
    list(APPEND cpprestsdk_CMAKE_ARGS "-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON")
    list(APPEND cpprestsdk_CMAKE_ARGS "-DCPPREST_EXCLUDE_WEBSOCKETS:BOOL=ON")
    list(APPEND cpprestsdk_CMAKE_ARGS "-DBUILD_SHARED_LIBS:BOOL=OFF")
    set(cpprestsdk_PATCH_SOURCE_FILEPATH "${CMAKE_SOURCE_DIR}/third_party/cmake/cpprestsdk.patch")
    ExternalProject_Add(
        cpprestsdk
        URL "https://github.com/microsoft/cpprestsdk/archive/refs/tags/2.10.18.tar.gz"
        URL_HASH "SHA256=6bd74a637ff182144b6a4271227ea8b6b3ea92389f88b25b215e6f94fd4d41cb"
        PATCH_COMMAND patch -p0 < ${cpprestsdk_PATCH_SOURCE_FILEPATH}
        CMAKE_ARGS ${cpprestsdk_CMAKE_ARGS}
        INSTALL_DIR ${cpprestsdk_INSTALL_PREFIX}
        GIT_PROGRESS TRUE
        USES_TERMINAL_DOWNLOAD TRUE
        DEPENDS BoringSSL
    )

    ExternalProject_Get_Property(cpprestsdk INSTALL_DIR)
    set(cpprestsdk_DIR "${INSTALL_DIR}/lib/cmake/cpprestsdk")
    unset(SOURCE_DIR)
    list(APPEND GXF_SUPERBUILD_DEPENDS "cpprestsdk")
endif()

list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-Dcpprestsdk_DIR:PATH=${cpprestsdk_DIR}")
