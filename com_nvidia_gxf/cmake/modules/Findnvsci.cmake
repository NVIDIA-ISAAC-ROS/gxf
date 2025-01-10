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

if(NOT nvsci_DIR)
    message(FATAL_ERROR "Must provide a value for nvsci_DIR")
endif()

set(NVSCI_SHARED_LIBS
    nvscibuf
    nvscisync
    nvscievent
    nvscicommon
    nvsciipc
    nvos
)
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    list(APPEND NVSCI_SHARED_LIBS "gnat-23")
    list(APPEND NVSCI_SHARED_LIBS "nvsci_mm")
endif()

set(NVSCI_HEADERS
    nvscibuf.h
    nvscierror.h
    nvscievent.h
    nvsciipc.h
    nvscistream_api.h
    nvscistream.h
    nvscistream_types.h
    nvscisync.h
)

foreach(_header ${NVSCI_HEADERS})
    find_file(
        _header_path
        "${_header}"
        HINTS
            "${nvsci_DIR}/include"
            "${nvsci_DIR}/usr/include"
        REQUIRED
    )
endforeach()
get_filename_component(nvsci_INCLUDE_DIR ${_header_path} DIRECTORY)

set(nvsci_TARGETS "")
foreach(_libname ${NVSCI_SHARED_LIBS})
    find_file(
        ${_libname}_FILEPATH
        "lib${_libname}.so"
        HINTS
            "${nvsci_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu"
            "${nvsci_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/tegra"
            "${nvsci_DIR}/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu"
        REQUIRED
    )
    add_library(nvsci_${_libname} SHARED IMPORTED)
    set_target_properties(nvsci_${_libname}
        PROPERTIES
        IMPORTED_LOCATION ${${_libname}_FILEPATH}
        INTERFACE_INCLUDE_DIRECTORIES ${nvsci_INCLUDE_DIR}
    )
    add_library(nvsci::${_libname} ALIAS nvsci_${_libname})
    list(APPEND nvsci_TARGETS "nvsci::${_libname}")
endforeach()
get_filename_component(nvsci_INSTALL_LIB_DIR ${_libname}_FILEPATH DIRECTORY)

add_library(nvsci INTERFACE)
target_link_libraries(nvsci INTERFACE ${nvsci_TARGETS})
add_library(nvsci::nvsci ALIAS nvsci)

set(nvsci_FOUND TRUE)
