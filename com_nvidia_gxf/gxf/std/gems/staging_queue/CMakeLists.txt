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

set(STD_GEMS_STAGING_QUEUE_SUFFIXES
    staging_queue
    staging_queue_iterator
)

add_library(std_gems_staging_queue_staging_queue_iterator INTERFACE)
set_target_properties(std_gems_staging_queue_staging_queue_iterator
    PROPERTIES PUBLIC_HEADER "staging_queue_iterator.hpp"
)

add_library(std_gems_staging_queue_staging_queue INTERFACE)
set_target_properties(std_gems_staging_queue_staging_queue
    PROPERTIES PUBLIC_HEADER "staging_queue.hpp"
)
target_link_libraries(std_gems_staging_queue_staging_queue
    INTERFACE
        std_gems_staging_queue_staging_queue_iterator
        GXF::common
        Threads::Threads
)

# Export/install

set(STD_GEMS_STAGING_QUEUE_TARGET_NAMES "")
foreach(_suffix ${STD_GEMS_STAGING_QUEUE_SUFFIXES})
    set(_target_name std_gems_staging_queue_${_suffix})
    list(APPEND STD_GEMS_STAGING_QUEUE_TARGET_NAMES "${_target_name}")
    add_library(GXF::std::gems::staging_queue::${_suffix} ALIAS ${_target_name})
endforeach()

## Convenience aliases

add_library(std_gems_staging_queue INTERFACE)
target_link_libraries(std_gems_staging_queue
    INTERFACE ${STD_GEMS_STAGING_QUEUE_TARGET_NAMES}
)
add_library(GXF::std::gems::staging_queue ALIAS std_gems_staging_queue)

# Export/install

install(
    TARGETS ${STD_GEMS_STAGING_QUEUE_TARGET_NAMES};std_gems_staging_queue
    EXPORT gxfCoreTargets
    PUBLIC_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gxf/std/gems/staging_queue
    COMPONENT Core
)
