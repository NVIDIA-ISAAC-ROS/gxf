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

# lss is a transitive dependency for GXF required for building Breakpad.
# It is not referenced directly by GXF.

if(lss_DIR OR lss_ROOT)
    find_package(lss REQUIRED)
else()
    set(lss_HEADER "linux_syscall_support.h")
    set(lss_INSTALL_DIR "${CMAKE_BINARY_DIR}/lss-install")
    set(lss_PATCH_SOURCE_FILEPATH "${CMAKE_SOURCE_DIR}/third_party/lss_gcc.patch")

    ExternalProject_Add(
        lss_content
        URL "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/external/linux-syscall-support-93426bda6535943ff1525d0460aab5cc0870ccaf.tar.gz"
        URL_HASH "SHA256=6d2e98e9d360797db6348ae725be901c1947e5736d87f07917c2bd835b03eeef"
        PATCH_COMMAND git apply ${lss_PATCH_SOURCE_FILEPATH}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND cmake -E copy <SOURCE_DIR>/${lss_HEADER} "${lss_INSTALL_DIR}/third_party/lss/${lss_HEADER}"
    )

    add_library(lss INTERFACE)
    target_include_directories(lss
        INTERFACE "${lss_INSTALL_DIR}"
    )
    add_dependencies(lss lss_content)
endif()
