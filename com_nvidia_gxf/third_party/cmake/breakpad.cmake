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

if(breakpad_DIR OR breakpad_ROOT)
    find_package(breakpad REQUIRED)
else()
    get_target_property(lss_INCLUDE lss INTERFACE_INCLUDE_DIRECTORIES)

    # breakpad does not support CMake inherently and does not allow setting a custom install directory.
    # For GXF build purposes we will build in-place in the source directory using the
    # project's Make structure and provide Findbreakpad.cmake in the project.
    ExternalProject_Add(
        breakpad
        URL "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/external/breakpad-v2023.01.27.tar.gz"
        URL_HASH "SHA256=f187e8c203bd506689ce4b32596ba821e1e2f034a83b8e07c2c635db4de3cc0b"
        PATCH_COMMAND ${CMAKE_COMMAND} -E copy_directory ${lss_INCLUDE}/third_party <SOURCE_DIR>/src/third_party
        CONFIGURE_COMMAND cd <SOURCE_DIR> && <SOURCE_DIR>/configure
        BUILD_COMMAND cd <SOURCE_DIR> && make
        INSTALL_COMMAND ""
        DEPENDS lss
    )

    ExternalProject_Get_Property(breakpad SOURCE_DIR)
    set(breakpad_DIR "${SOURCE_DIR}/src")
    unset(SOURCE_DIR)

    list(APPEND GXF_SUPERBUILD_DEPENDS "breakpad")
endif()

list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-Dbreakpad_DIR:PATH=${breakpad_DIR}")
