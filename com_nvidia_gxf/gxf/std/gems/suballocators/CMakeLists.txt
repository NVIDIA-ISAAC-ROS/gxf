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

set(STD_GEMS_SUBALLOCATORS_SUFFIXES first_fit_allocator_base
                                    first_fit_allocator)

add_library(std_gems_suballocators_first_fit_allocator_base OBJECT
            first_fit_allocator_base.cpp)
target_include_directories(
  std_gems_suballocators_first_fit_allocator_base
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}>)
set_target_properties(std_gems_suballocators_first_fit_allocator_base
                      PROPERTIES PUBLIC_HEADER "first_fit_allocator_base.hpp")

add_library(std_gems_suballocators_first_fit_allocator INTERFACE)
set_target_properties(std_gems_suballocators_first_fit_allocator
                      PROPERTIES PUBLIC_HEADER "first_fit_allocator.hpp")
target_link_libraries(
  std_gems_suballocators_first_fit_allocator
  INTERFACE std_gems_suballocators_first_fit_allocator_base GXF::common)

# Export/install

set(STD_GEMS_SUBALLOCATORS_TARGET_NAMES "")
foreach(_suffix ${STD_GEMS_SUBALLOCATORS_SUFFIXES})
  set(_target_name std_gems_suballocators_${_suffix})
  list(APPEND STD_GEMS_SUBALLOCATORS_TARGET_NAMES "${_target_name}")
  add_library(GXF::std::gems::suballocators::${_suffix} ALIAS ${_target_name})
endforeach()

# Convenience aliases

add_library(std_gems_suballocators INTERFACE)
target_link_libraries(std_gems_suballocators
                      INTERFACE ${STD_GEMS_SUBALLOCATORS_TARGET_NAMES})
add_library(GXF::std::gems::suballocators ALIAS std_gems_suballocators)

# Export/install

install(
  TARGETS ${STD_GEMS_SUBALLOCATORS_TARGET_NAMES};std_gems_suballocators
  EXPORT gxfCoreTargets
  PUBLIC_HEADER
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gxf/std/gems/suballocators
    COMPONENT Core)
