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


# Package _only_ the base logger utility in `libgxf_logger.so`

add_library(logger SHARED
    logger.cpp
)
set_target_properties(logger PROPERTIES
    PUBLIC_HEADER "logger.hpp"
    OUTPUT_NAME "gxf_logger"
)
target_include_directories(logger
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(logger
    PRIVATE Threads::Threads
)
add_library(GXF::logger ALIAS logger)

add_library(gxf_logger OBJECT gxf_logger.cpp)
set_target_properties(gxf_logger PROPERTIES
    PUBLIC_HEADER "gxf_logger.hpp"
)
target_include_directories(gxf_logger
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(gxf_logger PUBLIC GXF::logger)
add_library(GXF::gxf_logger ALIAS gxf_logger)

include(GNUInstallDirs)
install(
    TARGETS logger;gxf_logger
    EXPORT gxfCoreTargets
    PUBLIC_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gxf/logger
    ARCHIVE
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/logger
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/logger
    COMPONENT Core
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
