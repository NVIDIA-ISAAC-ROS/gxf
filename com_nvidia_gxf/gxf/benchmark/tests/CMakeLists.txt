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
    NAME "test_benchmark"
    SOURCES
        test_benchmark.cpp
        async_trigger_data_replayer.cpp
        benchmark_report_checker.cpp
    PUBLIC_DEPENDS
        GXF::benchmark
        GXF::serialization
        GXF::std
        nlohmann_json::nlohmann_json
    NO_INSTALL
)

set(GXF_BENCHMARK_TEST_EXTENSIONS
    std
    sample
    serialization
    test_extension
    benchmark
    test_benchmark
)

set(GXF_BENCHMARK_TEST_APPS
    test_benchmark_allocator.yaml
    test_benchmark_data_replay_control.yaml
    test_benchmark_one_source_one_sink.yaml
    test_benchmark_one_source_one_sink_resource.yaml
    test_benchmark_one_source_two_sinks.yaml
    test_benchmark_two_sources_two_sinks.yaml
    test_benchmark_no_buffer.yaml
)

gxf_add_gxe_tests(
    GROUP_NAME benchmark_yaml_test
    APP_FILES ${GXF_BENCHMARK_TEST_APPS}
    EXT_DEPENDS ${GXF_BENCHMARK_TEST_EXTENSIONS}
)
