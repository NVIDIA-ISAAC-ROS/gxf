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

if(nvsci_DIR)
    find_package(nvsci REQUIRED)
else()
    if(NOT nvsci_URL)
        message(FATAL_ERROR "Selected preset does not provide nvsci URL source")
    endif()

    ExternalProject_Add(
        nvsci
        URL ${nvsci_URL}
        URL_HASH "SHA256=${nvsci_HASH}"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
    )
    set(nvsci_DIR "${CMAKE_BINARY_DIR}/nvsci-prefix/src/nvsci")
    list(APPEND GXF_SUPERBUILD_DEPENDS "nvsci")
endif()

list(APPEND GXF_SUPERBUILD_DEFINED_ARGS "-Dnvsci_DIR:PATH=${nvsci_DIR}")
