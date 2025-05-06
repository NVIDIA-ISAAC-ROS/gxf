# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

set(STD_GEMS_VIDEO_BUFFER_SUFFIXES video_buffer allocator)

add_library(std_gems_video_buffer_allocator INTERFACE)
target_include_directories(
  std_gems_video_buffer_allocator
  INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}>)
set_target_properties(std_gems_video_buffer_allocator
                      PROPERTIES PUBLIC_HEADER "allocator.hpp")
target_link_libraries(std_gems_video_buffer_allocator INTERFACE GXF::multimedia)

add_library(std_gems_video_buffer_video_buffer INTERFACE)
target_link_libraries(std_gems_video_buffer_video_buffer
                      INTERFACE std_gems_video_buffer_allocator)

# Export/install

set(STD_GEMS_VIDEO_BUFFER_TARGET_NAMES "")
foreach(_suffix ${STD_GEMS_VIDEO_BUFFER_SUFFIXES})
  set(_target_name std_gems_video_buffer_${_suffix})
  list(APPEND STD_GEMS_VIDEO_BUFFER_TARGET_NAMES "${_target_name}")
  add_library(GXF::std::gems::video_buffer::${_suffix} ALIAS ${_target_name})
endforeach()

# Convenience aliases

add_library(std_gems_video_buffer INTERFACE)
target_link_libraries(std_gems_video_buffer
                      INTERFACE ${STD_GEMS_VIDEO_BUFFER_TARGET_NAMES})
add_library(GXF::std::gems::video_buffer ALIAS std_gems_video_buffer)

# Export/install

install(
  TARGETS ${STD_GEMS_VIDEO_BUFFER_TARGET_NAMES};std_gems_video_buffer
  EXPORT gxfCoreTargets
  PUBLIC_HEADER
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gxf/std/gems/video_buffer
    COMPONENT Core)
