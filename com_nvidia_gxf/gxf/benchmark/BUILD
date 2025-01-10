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
)

load("//gxf:pybind.bzl", "pybind_library")

exports_files([
    "__init__.py",
])

py_library(
    name = "benchmark_extension",
    srcs = [
        "__init__.py",
        "benchmark_buffer_publisher.py",
        "benchmark_buffer.py",
        "benchmark_controller.py",
        "benchmark_publisher.py",
        "benchmark_sink.py",
        "benchmark_utils.py",
        "Components.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/serialization:serialization_extension",
        "//gxf/std:std_extension",
    ],
)

nv_gxf_cc_extension(
    name = "benchmark",
    srcs = ["benchmark.cpp"],
    visibility = ["//visibility:public"],
    deps = [
        "allocator_metrics_calculator",
        "basic_metrics_calculator",
        "benchmark_allocator",
        "benchmark_allocator_sink",
        "benchmark_controller",
        "benchmark_publisher",
        "benchmark_sink",
        "dummy_message_generator",
        "entity_buffer",
        "performance_calculator_base",
        "resource_profiler",
        "//gxf/benchmark/gems:data_replay_control",
    ],
    ext_deps = [
        "//gxf/serialization",
        "//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "benchmark_controller",
    srcs = ["benchmark_controller.cpp"],
    hdrs = ["benchmark_controller.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "benchmark_publisher",
        "benchmark_sink",
        "benchmark_allocator_sink",
        "performance_calculator_base",
        "resource_profiler_base",
        "//gxf/benchmark/gems:data_replay_control",
        "//gxf/serialization",
        "//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "entity_buffer",
    srcs = ["entity_buffer.cpp"],
    hdrs = ["entity_buffer.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "benchmark_publisher",
    srcs = ["benchmark_publisher.cpp"],
    hdrs = ["benchmark_publisher.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "entity_buffer",
        "//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "benchmark_sink",
    srcs = ["benchmark_sink.cpp"],
    hdrs = [
        "benchmark_sink.hpp",
        "benchmark_sink_base.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "benchmark_publisher",
        "performance_calculator_base",
        "//gxf/std",
        "@nlohmann-json",
    ],
)

nv_gxf_cc_library(
    name = "benchmark_allocator",
    srcs = ["benchmark_allocator.cpp"],
    hdrs = ["benchmark_allocator.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf",
        "//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "benchmark_allocator_sink",
    srcs = ["benchmark_allocator_sink.cpp"],
    hdrs = [
        "benchmark_allocator_sink.hpp",
        "benchmark_sink_base.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "benchmark_allocator",
        "benchmark_publisher",
        "performance_calculator_base",
        "//gxf/std",
        "@nlohmann-json",
    ],
)

nv_gxf_cc_library(
    name = "allocator_metrics_calculator",
    srcs = ["allocator_metrics_calculator.cpp"],
    hdrs = ["allocator_metrics_calculator.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "performance_calculator_base",
    ],
)

nv_gxf_cc_library(
    name = "performance_calculator_base",
    srcs = ["performance_calculator_base.cpp"],
    hdrs = ["performance_calculator_base.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/std",
        "@nlohmann-json",
    ],
)

nv_gxf_cc_library(
    name = "basic_metrics_calculator",
    srcs = ["basic_metrics_calculator.cpp"],
    hdrs = ["basic_metrics_calculator.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "performance_calculator_base",
        "//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "resource_profiler_base",
    srcs = ["resource_profiler_base.cpp"],
    hdrs = ["resource_profiler_base.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "performance_calculator_base",
        "//gxf/std",
    ]
)

nv_gxf_cc_library(
    name = "dummy_message_generator",
    srcs = ["dummy_message_generator.cpp"],
    hdrs = ["dummy_message_generator.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//common:nvtx_helper",
        "//gxf/benchmark/gems:dummy_message",
        "//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "resource_profiler",
    srcs = ["resource_profiler.cpp"],
    hdrs = ["resource_profiler.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "resource_profiler_base",
        "//gxf/std",
    ],
)

load("//registry/build:registry.bzl", "register_extension")

exports_files(["LICENSE"])

register_extension(
    name = "register_benchmark_ext",
    badges = [""],
    extension = "benchmark",
    namespace="gxf",
    headers = [
        "allocator_metrics_calculator.hpp",
        "basic_metrics_calculator.hpp",
        "benchmark_controller.hpp",
        "benchmark_publisher.hpp",
        "benchmark_sink.hpp",
        "benchmark_sink_base.hpp",
        "dummy_message_generator.hpp",
        "entity_buffer.hpp",
        "performance_calculator_base.hpp",
        "resource_profiler_base.hpp",
    ],
    python_sources = [
        "//gxf/benchmark:__init__.py",
        "//gxf/benchmark:Components.py",
    ],
    labels = ["gxf"],
    license_file = ":LICENSE",
    local_dependencies = [
        "//gxf/std:register_std_ext",
        "//gxf/serialization:register_serialization_ext",
    ],
    priority = "1",
    url = "www.nvidia.com",
    uuid = "b53ebe0c-e427-4485-8ca2-a4ddc8664bec",
    version = "0.0.1",
    visibility = ["//visibility:public"],
)
