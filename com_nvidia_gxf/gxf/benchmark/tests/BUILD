"""
 SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

load(
    "//gxf:gxf.bzl",
    "nv_gxf_cc_extension",
    "nv_gxf_cc_library",
    "nv_gxf_cc_test_group",
    "nv_gxf_test_app",
)

load("//gxf/python:pygxf.bzl", "nv_pygxf_test")

nv_pygxf_test(
    name = "test_benchmark_python_apis",
    srcs = ["test_benchmark_python_apis.py"],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/serialization",
        "//gxf/benchmark",
    ],
    deps = [
        "//gxf/std:std_extension",
        "//gxf/benchmark:benchmark_extension",
        "//gxf/sample:sample_extension"
    ],
    tags = ["pytest"],
)

# FAILS intermittently in CI
# nv_gxf_test_app(
#     name = "benchmark_one_source_one_sink",
#     srcs = ["test_benchmark_one_source_one_sink.yaml"],
#     extensions = [
#         "//gxf/sample",
#         "//gxf/serialization",
#         "//gxf/std",
#         "//gxf/test/extensions:test",
#         "//gxf/benchmark:benchmark",
#         "test_benchmark",
#     ],
#     deps = ["//gxf/test:test_main"],
#     tags = ["manual"],
# )

nv_gxf_cc_test_group(
    name = "tests",
    apps = [
        "gxf/benchmark/tests/test_benchmark_allocator.yaml",
        "gxf/benchmark/tests/test_benchmark_data_replay_control.yaml",
        "gxf/benchmark/tests/test_benchmark_one_source_one_sink_resource.yaml",
        "gxf/benchmark/tests/test_benchmark_one_source_two_sinks.yaml",
        "gxf/benchmark/tests/test_benchmark_two_sources_two_sinks.yaml",
        "gxf/benchmark/tests/test_benchmark_no_buffer.yaml",
    ],
    extensions = [
        "//gxf/sample",
        "//gxf/serialization",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/benchmark:benchmark",
        "test_benchmark",
    ],
    manifest_path_hack = "gxf/benchmark/tests/",
    deps = ["//gxf/test:test_main"],
)

nv_gxf_cc_extension(
    name = "test_benchmark",
    srcs = ["test_benchmark.cpp"],
    deps = [
        "async_trigger_data_replayer",
        "benchmark_report_checker",
        "//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "benchmark_report_checker",
    srcs = ["benchmark_report_checker.cpp"],
    hdrs = ["benchmark_report_checker.hpp"],
    deps = [
        "//gxf/benchmark:basic_metrics_calculator",
        "//gxf/serialization",
        "//gxf/std",
        "@nlohmann-json",
    ],
)

nv_gxf_cc_library(
    name = "async_trigger_data_replayer",
    srcs = ["async_trigger_data_replayer.cpp"],
    hdrs = ["async_trigger_data_replayer.hpp"],
    deps = [
        "//gxf/std",
        "//gxf/benchmark/gems:data_replay_control",
    ],
)

load("//registry/build:registry.bzl", "register_extension")

register_extension(
    name = "register_test_benchmark_ext",
    badges = [""],
    extension = "test_benchmark",
    namespace="gxf",
    headers = [
        "benchmark_report_checker.hpp",
    ],
    labels = ["gxf"],
    local_dependencies = [
        "//gxf/benchmark:register_benchmark_ext",
        "//gxf/std:register_std_ext",
    ],
    priority = "1",
    url = "www.nvidia.com",
    uuid = "fffbb93f-01dd-41d1-a2a6-95c4530b044e",
    version = "0.0.1",
    visibility = ["//visibility:public"],
)
