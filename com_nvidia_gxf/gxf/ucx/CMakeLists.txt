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
    NAME ucx
    SOURCES
        ucx_common.cpp
        ucx_component_serializer.cpp
        ucx_context.cpp
        ucx_entity_serializer.cpp
        ucx_extension.cpp
        ucx_receiver.cpp
        ucx_serialization_buffer.cpp
        ucx_transmitter.cpp
    PUBLIC_HEADERS
        ucx_common.hpp
        ucx_component_serializer.hpp
        ucx_context.hpp
        ucx_entity_serializer.hpp
        ucx_receiver.hpp
        ucx_serialization_buffer.hpp
        ucx_transmitter.hpp
    PUBLIC_DEPENDS
        GXF::multimedia
        GXF::serialization
        GXF::std
        ucx::ucp
        ucx::ucs
)
target_link_options(ucx
    PRIVATE
        -Wno-unknown-pragmas
        -Wno-deprecated-declarations
)

# TODO: it is unclear how/whether ucx tests can be properly executed from a single process.
# More guidance is required before GXF UCX tests may be enabled.
# if(BUILD_TESTING)
#     add_subdirectory(tests)
# endif()
