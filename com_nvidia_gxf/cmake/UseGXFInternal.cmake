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

# Macros to help streamline GXF CMake infrastructure development.
include(UseGXF)

# Wrapper macro to add an extension library in the GXF core repository.
#
# Inputs: See `gxf_add_extension_library` for details.
# Byproducts:
# - See `gxf_add_extension_library` for additional details.
# - Appends the extension name to the GXF_EXTENSION_LIBRARY_TARGETS variable. This is used in
#   the GXF Core libraries build system to update the build manifest output file.
macro(gxf_core_add_extension_library)
    cmake_parse_arguments("GXF_CORE_EXT" "NO_INSTALL" "NAME" "" ${ARGN})
    gxf_add_extension_library("${ARGN}")

    if(NOT ${GXF_CORE_EXT_NO_INSTALL})
        set(GXF_EXTENSION_LIBRARY_TARGETS "${GXF_EXTENSION_LIBRARY_TARGETS};${GXF_CORE_EXT_NAME}")
        set(GXF_EXTENSION_LIBRARY_TARGETS "${GXF_EXTENSION_LIBRARY_TARGETS}" PARENT_SCOPE)
    endif()
endmacro()

# Wrapper macro to generate a YAML manifest file in the GXF core repository.
# See `gxf_generate_manifest_file` for details.
function(gxf_core_generate_manifest_file)
    cmake_parse_arguments("GXF_CORE_MANIFEST" "" "HEADER" "" ${ARGN})
    if(NOT "${GXF_CORE_MANIFEST_HEADER}")
        set(GXF_CORE_MANIFEST_HEADER
"# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the \"License\")\\\;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"
        )
    endif()
    gxf_generate_manifest_file(
        HEADER "${GXF_CORE_MANIFEST_HEADER}"
        "${ARGN}")
endfunction()

# Creates a custom command to copy files to an output folder at build time.
# Inputs:
# - _file: The short file name to copy
# - _source_dir: The source directory where the file is found
# - _target_dir: The output directory to which the file should be copied
function(gxf_copy_to_output _file _source_dir _target_dir)
    set(_source_file "${_source_dir}/${_file}")
    set(_output_file "${_target_dir}/${_file}")
    add_custom_command(
        OUTPUT ${_output_file}
        COMMAND ${CMAKE_COMMAND} -E copy "${_source_file}" "${_output_file}"
        COMMENT "Copying '${_file}' to ${_target_dir}"
    )
endfunction()

function(gxf_copy_to_output_dir)
    set(PREFIX "COPY")
    set(options ALL)
    set(oneValueArgs TARGET_NAME)
    set(multiValueArgs FILES)
    cmake_parse_arguments(${PREFIX} "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(output_files "")
    foreach(file ${COPY_FILES})
        gxf_copy_to_output(${file} "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")
        list(APPEND output_files "${CMAKE_CURRENT_BINARY_DIR}/${file}")
    endforeach()
    if(${COPY_ALL})
        add_custom_target(${COPY_TARGET_NAME}
            ALL
            DEPENDS ${output_files}
            COMMENT "Copying files"
        )
    else()
        add_custom_target(${COPY_TARGET_NAME}
            DEPENDS ${output_files}
            COMMENT "Copying files"
        )
    endif()
endfunction()

# Add an executable with GoogleTest entries for testing GXF extensions.
function(gxf_add_gtests)
    set(PREFIX "TEST")
    set(oneValueArgs EXT_NAME)
    set(multiValueArgs SOURCES DATA_FILES DEPENDS BUILD_DEPENDS)
    cmake_parse_arguments(${PREFIX} "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(target_name "GXF${TEST_EXT_NAME}Test")
    add_executable(${target_name} ${TEST_SOURCES})
    target_link_libraries(${target_name} PRIVATE ${TEST_DEPENDS})
    gtest_add_tests(
        TARGET ${target_name}
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        SOURCES "${TEST_SOURCES}"
    )
    gxf_copy_to_output_dir(
        TARGET_NAME "copy_${target_name}_data_files"
        FILES ${TEST_DATA_FILES}
        ALL
    )
    add_dependencies(${target_name}
        copy_${target_name}_data_files
        ${TEST_BUILD_DEPENDS}
    )
endfunction()


# Add a group of GXF extension YAML app tests that share a common set of extension dependencies.
function(gxf_add_gxe_tests)
    set(PREFIX "TEST")
    set(multiValueArgs GROUP_NAME APP_FILES EXT_DEPENDS)
    cmake_parse_arguments(${PREFIX} "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    gxf_copy_to_output_dir(
        TARGET_NAME "copy_${TEST_GROUP_NAME}_to_output"
        FILES ${TEST_APP_FILES}
        ALL
    )
    gxf_core_generate_manifest_file(
        EXTENSION_TARGETS "${TEST_EXT_DEPENDS}"
    )

    foreach(app_file ${TEST_APP_FILES})
        STRING(REGEX REPLACE ".*/" "" TEST_NAME ${app_file})
        STRING(REGEX REPLACE ".yaml" "" TEST_NAME ${TEST_NAME})
        add_test(
            NAME "${TEST_NAME}"
            COMMAND $<TARGET_FILE:gxe>
                -app ${CMAKE_CURRENT_BINARY_DIR}/${app_file}
                -manifest ${CMAKE_CURRENT_BINARY_DIR}/manifest.yaml
        )
    endforeach()

    add_custom_target(
        ${TEST_GROUP_NAME}
        DEPENDS
            copy_${TEST_GROUP_NAME}_to_output
            gxe
            ${TEST_EXT_DEPENDS}
    )
endfunction()
