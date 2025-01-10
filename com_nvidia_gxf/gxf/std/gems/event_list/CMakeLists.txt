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

set(STD_GEMS_EVENT_LIST_SUFFIXES
    event_list
    unique_event_list
)

add_library(std_gems_event_list_event_list INTERFACE)
set_target_properties(std_gems_event_list_event_list
    PROPERTIES PUBLIC_HEADER "event_list.hpp"
)
target_link_libraries(std_gems_event_list_event_list
    INTERFACE
        GXF::common
        Threads::Threads
)

add_library(std_gems_event_list_unique_event_list INTERFACE)
set_target_properties(std_gems_event_list_unique_event_list
    PROPERTIES PUBLIC_HEADER "unique_event_list.hpp"
)
target_link_libraries(std_gems_event_list_unique_event_list
    INTERFACE
        GXF::common
        Threads::Threads
)

## Convenience alias

set(STD_GEMS_EVENT_LIST_TARGET_NAMES "")
foreach(_suffix ${STD_GEMS_EVENT_LIST_SUFFIXES})
    set(_target_name std_gems_event_list_${_suffix})
    list(APPEND STD_GEMS_EVENT_LIST_TARGET_NAMES "${_target_name}")
    add_library(GXF::std::gems::event_list::${_suffix} ALIAS ${_target_name})
endforeach()

add_library(std_gems_event_list INTERFACE)
target_link_libraries(std_gems_event_list
    INTERFACE ${STD_GEMS_EVENT_LIST_TARGET_NAMES}
)
add_library(GXF::std::gems::event_list ALIAS std_gems_event_list)

## Export/install

install(
    TARGETS ${STD_GEMS_EVENT_LIST_TARGET_NAMES};std_gems_event_list
    EXPORT gxfCoreTargets
    PUBLIC_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gxf/std/gems/event_list
    COMPONENT Core
)
