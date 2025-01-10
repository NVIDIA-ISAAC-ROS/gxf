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

set(app_SOURCES
    application.cpp
    arg_parse.cpp
    config_parser.cpp
    driver.cpp
    entity_group.cpp
    extension_manager.cpp
    graph_entity.cpp
    segment.cpp
    worker.cpp
)
set(app_HEADERS
    application.hpp
    arg_parse.hpp
    arg.hpp
    config_parser.hpp
    driver.hpp
    entity_group.hpp
    extension_manager.hpp
    graph_entity.hpp
    graph_utils.hpp
    proxy_component.hpp
    segment.hpp
    worker.hpp
)

set(app_INSTALL_RPATH
    "$ORIGIN"
    "$ORIGIN/../core"
    "$ORIGIN/../std"
    "$ORIGIN/../common"
    "$ORIGIN/../http"
    "$ORIGIN/../ucx"
)
string(REPLACE ";" ":" app_INSTALL_RPATH "${app_INSTALL_RPATH}")

add_library(app SHARED ${app_SOURCES})
set_target_properties(app
    PROPERTIES
    PUBLIC_HEADER "${app_HEADERS}"
    OUTPUT_NAME "gxf_app"
    INSTALL_RPATH "${app_INSTALL_RPATH}"
)
target_include_directories(app
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(app
    PUBLIC
        GXF::common
        GXF::core
        GXF::std
        yaml-cpp::yaml-cpp
    PRIVATE
        breakpad::breakpad
        breakpad::breakpad_client
        GXF::http
        GXF::ucx
        ${CMAKE_DL_LIBS}
)
add_library(GXF::app ALIAS app)

install(
    TARGETS app
    EXPORT gxfExtensionTargets
    PUBLIC_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gxf/app
    ARCHIVE
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/app
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/app
    COMPONENT App
)

if(BUILD_TESTING)
    add_subdirectory(tests)
    add_subdirectory(sample)
endif()
