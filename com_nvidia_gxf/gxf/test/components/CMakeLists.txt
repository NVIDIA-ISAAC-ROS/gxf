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

set(GXF_TEST_COMPONENTS_PREFIXES
    entity_monitor
    mock_allocator
    mock_codelet
    mock_failure
    mock_receiver
    mock_transmitter
    tensor_comparator
    tensor_generator
)

set(GXF_TEST_COMPONENTS_SOURCES "")
set(GXF_TEST_COMPONENTS_HEADERS "")
foreach(_prefix ${GXF_TEST_COMPONENTS_PREFIXES})
    list(APPEND GXF_TEST_COMPONENTS_SOURCES ${_prefix}.cpp)
    list(APPEND GXF_TEST_COMPONENTS_HEADERS ${_prefix}.hpp)
endforeach()

add_library(test_components STATIC ${GXF_TEST_COMPONENTS_SOURCES})
set_target_properties(test_components
    PROPERTIES PUBLIC_HEADER "${GXF_TEST_COMPONENTS_HEADERS}"
)
target_include_directories(test_components
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(test_components
    PUBLIC GXF::std
    PRIVATE CUDA::cudart
)
add_library(GXF::test::components ALIAS test_components)

install(
    TARGETS test_components
    EXPORT gxfExtensionTargets
    PUBLIC_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gxf/test/components
    ARCHIVE
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/test/components
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/test/components
    COMPONENT Test
)
