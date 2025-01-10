
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

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(GXF_COMPONENTS "app;core;logger;${GXF_EXTENSION_LIBRARY_TARGETS}")
configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/GXFConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/GXFConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/GXF"
)

# Generates a file to set the GXF version on import in a downstream project.
# gxf_VERSION is set by `project(gxf ...)` in the top level CMakeLists.txt.
# https://cmake.org/cmake/help/latest/module/CMakePackageConfigHelpers.html#generating-a-package-version-file
write_basic_package_version_file(
    GXFConfigVersion.cmake
    VERSION ${gxf_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/GXFConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/GXFConfigVersion.cmake"
    DESTINATION
        "${CMAKE_INSTALL_LIBDIR}/cmake/GXF"
)
