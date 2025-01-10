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

add_library(std_gems_pool_fixed_pool_uint64 INTERFACE)
set_target_properties(std_gems_pool_fixed_pool_uint64
    PROPERTIES PUBLIC_HEADER "fixed_pool_uint64.hpp"
)
target_link_libraries(std_gems_pool_fixed_pool_uint64
    INTERFACE GXF::common
)
add_library(GXF::std::gems::pool::fixed_pool_uint64 ALIAS std_gems_pool_fixed_pool_uint64)

install(
    TARGETS std_gems_pool_fixed_pool_uint64
    EXPORT gxfCoreTargets
    PUBLIC_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gxf/std/gems/pool
    COMPONENT Core
)

add_library(GXF::std::gems::pool ALIAS std_gems_pool_fixed_pool_uint64)
