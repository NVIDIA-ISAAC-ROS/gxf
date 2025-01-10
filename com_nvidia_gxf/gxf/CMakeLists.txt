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

set(GXF_EXTENSION_LIBRARY_TARGETS "")

# "GXF Core" components require both libgxf_core.so and libgxf_std.so
add_subdirectory(core)
add_subdirectory(logger)
add_subdirectory(std)

# Additional extensions
add_subdirectory(app)
add_subdirectory(test) # include `libgxf_test.so` in the standard GXF binary distribution
add_subdirectory(cuda)
add_subdirectory(multimedia)
add_subdirectory(ipc)
add_subdirectory(sample)
add_subdirectory(stream)
add_subdirectory(serialization)
add_subdirectory(ucx) # depends on serialization

add_subdirectory(behavior_tree) # depends on serialization
add_subdirectory(npp)
add_subdirectory(network)
add_subdirectory(python_codelet)

add_subdirectory(benchmark)
add_subdirectory(rmm)

# `gxe` subdirectory must be processed last so that
# engine manifest is fully populated with extensions
add_subdirectory(gxe)

# TODO: JIRA NVG-3105
# Build remaining libraries with CMake
# add_subdirectory(cli)
# add_subdirectory(python)

install(
    EXPORT gxfCoreTargets
    FILE GXFCoreTargets.cmake
    NAMESPACE GXF::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GXF
)

install(
    EXPORT gxfExtensionTargets
    FILE GXFExtensionTargets.cmake
    NAMESPACE GXF::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GXF
)

set(GXF_EXTENSION_LIBRARY_TARGETS ${GXF_EXTENSION_LIBRARY_TARGETS} PARENT_SCOPE)
