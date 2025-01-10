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

###################################################################
# c_api
###################################################################
set(GXF_CORE_C_API_HEADERS
    gxf.h
    gxf_ext.h
)

add_library(core_c_api INTERFACE)
set_target_properties(core_c_api
    PROPERTIES PUBLIC_HEADER "${GXF_CORE_C_API_HEADERS}"
)
add_library(GXF::core::c_api ALIAS core_c_api)

#######################################################################################
# Core (Internal)
#
# These sources do not depend on sources in the `std` directory.
# This "internal" target is required to manage the circular dependency among `core`
# and `std` directories.
#######################################################################################
set(GXF_CORE_INTERNAL_SOURCES
    parameter_registrar.cpp
    parameter_storage.cpp
    resource_manager.cpp
    type_registry.cpp
)
set(GXF_CORE_INTERNAL_PUBLIC_HEADERS
    common_expected_macro.hpp
    component.hpp
    entity_item.hpp
    entity.hpp
    expected.hpp
    expected_macro.hpp
    filepath.hpp
    handle.hpp
    parameter.hpp
    parameter_parser.hpp
    parameter_parser_std.hpp
    parameter_registrar.hpp
    parameter_storage.hpp
    parameter_wrapper.hpp
    registrar.hpp
    resource.hpp
    resource_manager.hpp
    resource_registrar.hpp
    type_registry.hpp
)

add_library(core_internal OBJECT ${GXF_CORE_INTERNAL_SOURCES})
set_target_properties(core_internal
    PROPERTIES PUBLIC_HEADER "${GXF_CORE_INTERNAL_PUBLIC_HEADERS}"
)
target_include_directories(core_internal
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(core_internal
    PUBLIC
        GXF::common
        GXF::core::c_api
        GXF::logger
        Threads::Threads
        magic_enum::magic_enum
    PRIVATE yaml-cpp::yaml-cpp
)
add_library(GXF::core_internal ALIAS core_internal)

#######################################################################################
# Core
#
# The `libgxf_core.so` library depends on sources in both `core` and `std` directories.
#######################################################################################
set(GXF_CORE_SOURCES
    gxf.cpp
    runtime.cpp
)
set(GXF_CORE_HEADERS
    runtime.hpp
)
add_library(core SHARED
    ${GXF_CORE_SOURCES}
    $<TARGET_OBJECTS:common_internal>
    $<TARGET_OBJECTS:common>
    $<TARGET_OBJECTS:gxf_logger>
    $<TARGET_OBJECTS:std_internal>
    $<TARGET_OBJECTS:std_gems_utils_internal>
)
set_target_properties(core
    PROPERTIES
        PUBLIC_HEADER "${GXF_CORE_HEADERS}"
        OUTPUT_NAME "gxf_core"
        INSTALL_RPATH "$ORIGIN:$ORIGIN/../logger:$ORIGIN/../std"
)
target_include_directories(core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(core
    PUBLIC
        GXF::common
        GXF::core_internal
        GXF::logger
        Threads::Threads
    PRIVATE
        GXF::std_internal
        yaml-cpp::yaml-cpp
)
add_library(GXF::core ALIAS core)

install(
    TARGETS core_c_api;core_internal;core
    EXPORT gxfCoreTargets
    PUBLIC_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gxf/core
    ARCHIVE
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/core
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/core
    COMPONENT Core
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
