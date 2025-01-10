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

set(STD_GEMS_UTILS_SOURCES time.cpp)
set(STD_GEMS_UTILS_HEADERS
    exponential_moving_average.hpp
    fast_running_median.hpp
    time.hpp
)

add_library(std_gems_utils_internal OBJECT ${STD_GEMS_UTILS_SOURCES})
set_target_properties(std_gems_utils_internal
    PROPERTIES PUBLIC_HEADER "${STD_GEMS_UTILS_HEADERS}"
)
target_include_directories(std_gems_utils_internal
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}>
)
target_link_libraries(std_gems_utils_internal
    PUBLIC
        GXF::common
        GXF::core_internal
)
add_library(GXF::std::gems::utils_internal ALIAS std_gems_utils_internal)

install(
    TARGETS std_gems_utils_internal
    EXPORT gxfCoreTargets
    PUBLIC_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gxf/std/gems/utils
    ARCHIVE
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/std/gems/utils
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/std/gems/utils
    COMPONENT Core
)
