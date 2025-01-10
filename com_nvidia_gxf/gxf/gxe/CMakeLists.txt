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

add_executable(gxe gxe.cpp)
target_link_libraries(gxe
    PRIVATE
        GXF::common
        GXF::core
        breakpad::breakpad
        breakpad::breakpad_client
        gflags::gflags_static
)
set_target_properties(gxe
    PROPERTIES
    INSTALL_RPATH "$ORIGIN:../lib/gxf/core"
)
set(GXF_EXTENSION_LIBRARY_TARGETS "${GXF_EXTENSION_LIBRARY_TARGETS};gxe" PARENT_SCOPE)

# Generate the `manifest.yaml` file to describe the GXF extension shared libraries
# that are available in the build output directory.
gxf_core_generate_manifest_file(EXTENSION_TARGETS "${GXF_EXTENSION_LIBRARY_TARGETS}")
# Some tests rely on the default manifest file, which includes all possible GXF Core extensions.
# Use the alias target to indicate that dependency to the CMake build system.
# TODO: refactor these tests to rely on a test-specific manifest with a minium list of extensions.
add_custom_target(gxf_core_manifest
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/manifest.yaml
        ${GXF_EXTENSION_LIBRARY_TARGETS}
)

install(
    TARGETS gxe
    EXPORT gxfCoreTargets
    RUNTIME
        DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT Engine
)

# Generate the install manifest from the build manifest.
# We assume that GXF extension shared libraries maintain a common subdirectory structure
# within either build or install parent directories.
install(CODE "file(READ ${CMAKE_CURRENT_BINARY_DIR}/manifest.yaml MANIFEST_STRING)")
install(CODE "string(REPLACE \"${CMAKE_BINARY_DIR}\" \"${CMAKE_INSTALL_LIBDIR}\" INSTALL_MANIFEST_STRING \"\${MANIFEST_STRING}\")")
install(CODE "file(WRITE ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/manifest.yaml \"\${INSTALL_MANIFEST_STRING}\")")
