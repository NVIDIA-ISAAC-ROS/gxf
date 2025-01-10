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

if(breakpad_DIR AND NOT breakpad_ROOT)
    set(breakpad_ROOT ${breakpad_DIR})
endif()
if(NOT breakpad_ROOT)
    message(FATAL_ERROR "Must supply either breakpad_ROOT or breakpad_DIR")
endif()

set(breakpad_INCLUDE_DIR ${breakpad_ROOT})
find_file(
    breakpad_STATIC_LIB
    NAMES libbreakpad.a
    PATHS
        "${breakpad_ROOT}"
        "${breakpad_ROOT}/lib"
    REQUIRED
)
find_file(
    breakpad_CLIENT_STATIC_LIB
    NAMES libbreakpad_client.a
    PATHS "${breakpad_ROOT}/client/linux"
    REQUIRED
)
find_file(
    breakpad_DISASM_STATIC_LIB
    NAMES libdisasm.a
    PATHS "${breakpad_ROOT}/third_party/libdisasm"
    REQUIRED
)

find_file(
    breakpad_exception_handler_header
    NAMES exception_handler.h
    PATHS
        "${breakpad_ROOT}/client/linux/handler"
        "${breakpad_ROOT}/include/client/linux/handler"
    REQUIRED
)

add_library(breakpad STATIC IMPORTED)
set_target_properties(breakpad
    PROPERTIES IMPORTED_LOCATION ${breakpad_STATIC_LIB}
)

add_library(breakpad_client STATIC IMPORTED)
set_target_properties(breakpad_client
    PROPERTIES IMPORTED_LOCATION ${breakpad_CLIENT_STATIC_LIB}
)

add_library(breakpad_disasm STATIC IMPORTED)
set_target_properties(breakpad_disasm
    PROPERTIES IMPORTED_LOCATION ${breakpad_DISASM_STATIC_LIB}
)

foreach(_target breakpad;breakpad_client;breakpad_disasm)
    target_include_directories("${_target}"
        INTERFACE
            $<BUILD_INTERFACE:${breakpad_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:"">
    )
    add_library(breakpad::${_target} ALIAS ${_target})
endforeach()

set(breakpad_FOUND TRUE)
