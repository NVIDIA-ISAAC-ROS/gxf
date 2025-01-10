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

# Helper functions and macros for working with GXF extensions.


# Creates a shared library target for a GXF extension library.
#
# `gxf_add_extension_library` is a convenient method to define a simple GXF extension library
# target in a CMake project.
#
# Inputs:
# - NAME: The name of the extension library
# - SOURCES: The list of source files for the extension library
# - PUBLIC_HEADERS: The list of public header files for the extension library
# - PUBLIC_DEPENDS: The list of public CMake target dependencies for the extension library
# - PRIVATE_DEPENDS: The list of private CMake target dependencies for the extension library
# Note: All input filepaths are relative to the directory in which this macro is invoked.
#
# Byproducts:
# - A shared library target "${NAME}". You can set properties on this target.
# - A namespaced alias target named "GXF::${NAME}". Use the namespace alias target for easier
#   debugging when linking into other library targets.
# - An installation component "${NAME}" for custom installation configuration with the CMake CLI.
function(gxf_add_extension_library)
    set(PREFIX "EXT")
    set(options NO_INSTALL)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES PUBLIC_HEADERS PUBLIC_DEPENDS PRIVATE_DEPENDS)
    cmake_parse_arguments(${PREFIX} "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_library(${EXT_NAME} SHARED "${EXT_SOURCES}")
    set_target_properties(${EXT_NAME}
        PROPERTIES
        OUTPUT_NAME "gxf_${EXT_NAME}"
        PUBLIC_HEADER "${EXT_PUBLIC_HEADERS}"
        INSTALL_RPATH "$ORIGIN:$ORIGIN/../core:$ORIGIN/../std"
    )
    target_include_directories(${EXT_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(${EXT_NAME}
        PUBLIC "${EXT_PUBLIC_DEPENDS}"
        PRIVATE "${EXT_PRIVATE_DEPENDS}"
    )
    add_library(GXF::${EXT_NAME} ALIAS ${EXT_NAME})

    if(NOT ${EXT_NO_INSTALL})
        install(
            TARGETS ${EXT_NAME}
            EXPORT gxfExtensionTargets
            PUBLIC_HEADER
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gxf/${EXT_NAME}
            ARCHIVE
                DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/${EXT_NAME}
            LIBRARY
                DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/${EXT_NAME}
            COMPONENT ${EXT_NAME}
        )
    endif()
endfunction()

# Generates a YAML manifest file describing GXF extension shared library locations.
#
# Inputs:
# - EXTENSION_TARGETS: List of extension library CMake target names to include in the manifest.
#   The GXE executable will load extensions in the order specified.
# - DESTINATION: Output folder for the manifest file. Defaults to the current build directory.
# - HEADER: Optional header string to include at the top of the manifest file.
#
# Byproducts:
# - Generates a manifest.yaml file in the specified DESTINATION folder describing extension
#   library target file locations.
function(gxf_generate_manifest_file)
    cmake_parse_arguments("GXF_MANIFEST" "" "DESTINATION;HEADER" "EXTENSION_TARGETS" ${ARGN})
    set(_extension_entries "")
    if(NOT GXF_MANIFEST_DESTINATION)
        set(GXF_MANIFEST_DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    endif()
    foreach(_extension ${GXF_MANIFEST_EXTENSION_TARGETS})
        string(APPEND _extension_entries "
- $<TARGET_FILE:${_extension}>")
    endforeach()
    set(_manifest_filename ${GXF_MANIFEST_DESTINATION}/manifest.yaml)
    file(GENERATE
        OUTPUT ${_manifest_filename}
        CONTENT
"${GXF_MANIFEST_HEADER}
extensions:${_extension_entries}
"
    )
endfunction()
