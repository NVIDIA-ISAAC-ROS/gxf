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

if(magic_enum_DIR OR magic_enum_ROOT)
    find_package(magic_enum REQUIRED)
else()
    set(magic_enum_CMAKE_ARGS "-DMAGIC_ENUM_OPT_INSTALL:BOOL=ON")
    list(APPEND magic_enum_CMAKE_ARGS "-DMAGIC_ENUM_OPT_BUILD_EXAMPLES:BOOL=OFF")
    list(APPEND magic_enum_CMAKE_ARGS "-DMAGIC_ENUM_OPT_BUILD_TESTS:BOOL=OFF")
    set(magic_enum_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/magic_enum-install")
    list(APPEND magic_enum_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX:PATH=${magic_enum_INSTALL_PREFIX}")

    ExternalProject_Add(
        magic_enum
        GIT_REPOSITORY "https://github.com/Neargye/magic_enum"
        GIT_TAG "v0.9.3"
        CMAKE_ARGS ${magic_enum_CMAKE_ARGS}
        INSTALL_DIR ${magic_enum_INSTALL_PREFIX}
    )
    ExternalProject_Get_Property(magic_enum INSTALL_DIR)
    set(magic_enum_DIR "${INSTALL_DIR}/lib/cmake/magic_enum")
    unset(INSTALL_DIR)

    list(APPEND GXF_SUPERBUILD_DEPENDS "magic_enum")
endif()

list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-Dmagic_enum_DIR:PATH=${magic_enum_DIR}")
