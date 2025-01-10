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
    NAME "benchmark"
    SOURCES
        allocator_metrics_calculator.cpp
        basic_metrics_calculator.cpp
        benchmark_allocator.cpp
        benchmark_allocator_sink.cpp
        benchmark_controller.cpp
        benchmark.cpp
        benchmark_publisher.cpp
        benchmark_sink.cpp
        dummy_message_generator.cpp
        entity_buffer.cpp
        performance_calculator_base.cpp
        resource_profiler_base.cpp
        resource_profiler.cpp
        gems/dummy_message.cpp
    PUBLIC_HEADERS
        allocator_metrics_calculator.hpp
        basic_metrics_calculator.hpp
        benchmark_allocator.hpp
        benchmark_allocator_sink.hpp
        benchmark_controller.hpp
        benchmark_publisher.hpp
        benchmark_sink_base.hpp
        benchmark_sink.hpp
        dummy_message_generator.hpp
        entity_buffer.hpp
        performance_calculator_base.hpp
        resource_profiler_base.hpp
        resource_profiler.hpp
        gems/data_replay_control.hpp
        gems/dummy_message.hpp
    PUBLIC_DEPENDS
        GXF::serialization
        GXF::std
        nlohmann_json::nlohmann_json
    NO_INSTALL
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()


