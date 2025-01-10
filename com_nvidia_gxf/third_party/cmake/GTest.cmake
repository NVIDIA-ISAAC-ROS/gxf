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

if(GTest_DIR OR GTest_ROOT)
    find_package(GTest REQUIRED)
else()
    set(GTest_INSTALL_DIR "${CMAKE_BINARY_DIR}/GTest-install")
    set(GTest_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX:PATH=${GTest_INSTALL_DIR}")
    list(APPEND GTest_CMAKE_ARGS "-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=${CMAKE_POSITION_INDEPENDENT_CODE}")
    list(APPEND GTest_CMAKE_ARGS "-DBUILD_SHARED_LIBS:BOOL=OFF")
    ExternalProject_Add(
        GTest
        GIT_REPOSITORY https://github.com/google/googletest
        GIT_TAG v1.13.0
        CMAKE_ARGS ${GTest_CMAKE_ARGS}
        INSTALL_DIR ${GTest_INSTALL_DIR}
        UPDATE_DISCONNECTED TRUE
    )
    ExternalProject_Get_Property(GTest INSTALL_DIR)
    set(GTest_DIR "${INSTALL_DIR}/lib/cmake/GTest")
    unset(INSTALL_DIR)
    list(APPEND GXF_SUPERBUILD_DEPENDS "GTest")
endif()

list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-DGTest_DIR:PATH=${GTest_DIR}")
