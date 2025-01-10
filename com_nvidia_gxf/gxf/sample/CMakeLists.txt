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
    NAME sample
    SOURCES
        multi_ping_rx.cpp
        ping_batch_rx.cpp
        ping_rx.cpp
        ping_rx_async.cpp
        ping_tx.cpp
        ping_tx_async.cpp
        sample.cpp
    PUBLIC_HEADERS
        hello_world.hpp
        multi_ping_rx.hpp
        ping_batch_rx.hpp
        ping_rx.hpp
        ping_rx_async.hpp
        ping_tx.hpp
        ping_tx_async.hpp
    PUBLIC_DEPENDS
        GXF::core
        GXF::std
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
