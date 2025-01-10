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

add_library(std_gems_timed_job_list INTERFACE)
set_target_properties(std_gems_timed_job_list
    PROPERTIES PUBLIC_HEADER "timed_job_list.hpp"
)
target_link_libraries(std_gems_timed_job_list
    INTERFACE GXF::core_internal
)
add_library(GXF::std::gems::timed_job_list ALIAS std_gems_timed_job_list)

install(
    TARGETS std_gems_timed_job_list
    EXPORT gxfCoreTargets
    PUBLIC_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gxf/std/gems/timed_job_list
    COMPONENT Core
)
