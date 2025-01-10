
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

gxf_core_add_extension_library(
    NAME behavior_tree
    SOURCES
        behavior_tree.cpp
        constant_behavior.cpp
        entity_count_failure_repeat_controller.cpp
        parallel_behavior.cpp
        repeat_behavior.cpp
        selector_behavior.cpp
        sequence_behavior.cpp
        switch_behavior.cpp
        timer_behavior.cpp
    PUBLIC_HEADERS
        constant_behavior.hpp
        entity_count_failure_repeat_controller.hpp
        parallel_behavior.hpp
        repeat_behavior.hpp
        selector_behavior.hpp
        sequence_behavior.hpp
        switch_behavior.hpp
        timer_behavior.hpp
    PUBLIC_DEPENDS
        GXF::common
        GXF::core
        GXF::std
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
