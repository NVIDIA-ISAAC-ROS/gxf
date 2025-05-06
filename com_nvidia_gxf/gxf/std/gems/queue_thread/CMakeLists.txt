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

add_library(std_gems_queue_thread_queue_thread INTERFACE)
set_target_properties(std_gems_queue_thread_queue_thread
                      PROPERTIES PUBLIC_HEADER "queue_thread.hpp")
target_link_libraries(std_gems_queue_thread_queue_thread INTERFACE GXF::common)
add_library(GXF::std::gems::queue_thread::queue_thread ALIAS
            std_gems_queue_thread_queue_thread)

# Convenience aliases
add_library(std_gems_queue_thread INTERFACE)
target_link_libraries(std_gems_queue_thread
                      INTERFACE std_gems_queue_thread_queue_thread)
add_library(GXF::std::gems::queue_thread ALIAS std_gems_queue_thread)

install(
  TARGETS std_gems_queue_thread_queue_thread std_gems_queue_thread
  EXPORT gxfCoreTargets
  PUBLIC_HEADER
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gxf/std/gems/queue_thread
    COMPONENT Core)
