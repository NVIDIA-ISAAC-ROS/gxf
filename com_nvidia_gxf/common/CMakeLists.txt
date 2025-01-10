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

############################################################
# The common_internal library represents `common` sources
# that do not depend on GXF sources.
############################################################

set(GXF_COMMON_INTERNAL_SOURCES
    backtrace.cpp
    type_name.cpp
)
set(GXF_COMMON_INTERNAL_HEADERS
    any.hpp
    backtrace.hpp
    byte.hpp
    endian.hpp
    memory_utils.hpp
    nvtx_helper.hpp
    optional.hpp
    singleton.hpp
    strong_type.hpp
    type_name.hpp
    type_name_gnuc.hpp
    type_utils.hpp
)
add_library(common_internal OBJECT ${GXF_COMMON_INTERNAL_SOURCES})
set_target_properties(common_internal PROPERTIES
    PUBLIC_HEADER "${GXF_COMMON_INTERNAL_HEADERS}"
)
target_include_directories(common_internal
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(common_internal INTERFACE CUDA::nvtx3)
add_library(GXF::common_internal ALIAS common_internal)

############################################################
# The common library represents `common` sources
# that depend on GXF sources.
############################################################
set(GXF_COMMON_SOURCES logger.cpp)
set(GXF_COMMON_HEADERS
    assert.hpp
    expected.hpp
    fixed_map.hpp
    fixed_string.hpp
    fixed_vector.hpp
    iterator.hpp
    logger.hpp
    span.hpp
    unique_index_map.hpp
    yaml_parser.hpp
)
add_library(common OBJECT
    ${GXF_COMMON_SOURCES}
    $<TARGET_OBJECTS:common_internal>
    $<TARGET_OBJECTS:gxf_logger>
)
set_target_properties(common PROPERTIES
    PUBLIC_HEADER "${GXF_COMMON_HEADERS}"
)
target_include_directories(common
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(common
    PUBLIC
        GXF::common_internal
        GXF::logger
    PRIVATE Threads::Threads
)
add_library(GXF::common ALIAS common)

include(GNUInstallDirs)
install(
    TARGETS common;common_internal
    EXPORT gxfCoreTargets
    PUBLIC_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/common
    ARCHIVE
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/common
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/common
    COMPONENT Core
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
