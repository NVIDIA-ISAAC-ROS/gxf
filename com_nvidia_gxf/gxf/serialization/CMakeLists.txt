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
    NAME serialization
    SOURCES
        component_serializer.cpp
        endpoint.cpp
        entity_recorder.cpp
        entity_replayer.cpp
        entity_serializer.cpp
        file.cpp
        file_stream.cpp
        serialization_buffer.cpp
        serialization.cpp
        std_component_serializer.cpp
        std_entity_id_serializer.cpp
        std_entity_serializer.cpp
    PUBLIC_HEADERS
        component_serializer.hpp
        endpoint.hpp
        entity_recorder.hpp
        entity_replayer.hpp
        entity_serializer.hpp
        file.hpp
        file_stream.hpp
        serialization_buffer.hpp
        std_component_serializer.hpp
        std_entity_id_serializer.hpp
        std_entity_serializer.hpp
        tid_hash.hpp
    PUBLIC_DEPENDS
        GXF::core
        GXF::std
        CUDA::cudart
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
