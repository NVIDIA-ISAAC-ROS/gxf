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

load( "//gxf:gxf.bzl", "nv_gxf_app",)
load("//gxf/python:pygxf.bzl", "nv_pygxf_app")

# Benchmark the framework based on dummy source and GreedyScheduler
# (by directly connecting a publisher and a sink)
# bazel run //extensions/benchmark/apps:blank_gs_bm
nv_gxf_app(
    name = "blank_gs_bm",
    gxf_file = "blank_gs_bm.yaml",
    extensions = [
        "//gxf/benchmark",
        "//gxf/sample",
        "//gxf/std",
    ],
    visibility = ["//visibility:public"],
)

# Benchmark the framework based on dummy source and MultiThreadScheduler
# (by directly connecting a publisher and a sink)
# bazel run //extensions/benchmark/apps:blank_mt_bm
nv_gxf_app(
    name = "blank_mt_bm",
    gxf_file = "blank_mt_bm.yaml",
    extensions = [
        "//gxf/benchmark",
        "//gxf/sample",
        "//gxf/std",
    ],
    visibility = ["//visibility:public"],
)

# Benchmark the framework based on dummy source and EventBasedScheduler
# (by directly connecting a publisher and a sink)
# bazel run //extensions/benchmark/apps:blank_eb_bm
nv_gxf_app(
    name = "blank_eb_bm",
    gxf_file = "blank_eb_bm.yaml",
    extensions = [
        "//gxf/benchmark",
        "//gxf/sample",
        "//gxf/std",
    ],
    visibility = ["//visibility:public"],
)

# Benchmark the framework based on dummy source and selected scheduler
# (by directly connecting a publisher and a sink)
# Supported variations:
#   bazel run //gxf/benchmark/apps:blank_bm -- --scheduler GreedyScheduler
#   bazel run //gxf/benchmark/apps:blank_bm -- --scheduler MultiThreadScheduler
#   bazel run //gxf/benchmark/apps:blank_bm -- --scheduler EventBasedScheduler
nv_pygxf_app(
    name = "blank_bm",
    srcs = ["blank_bm.py"],
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
)

nv_pygxf_app(
    name = "stress_test_bm",
    args = ["--scheduler EventBasedScheduler --num_parallel_codelets 5000 --worker_thread_number 4"],
    srcs = ["stress_test_bm.py"],
    main = "stress_test_bm.py",
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
)
# Benchmark GXF with intentional contention
# bazel run //gxf/benchmark/apps:<xxx>
nv_pygxf_app(
    name = "message_contention_1px1hx5c_6t_r_bm",
    args = ["--parallel_pipeline_size 1 --pipeline_length 1 --dummy_component_size 5 --worker_thread_number 6"],
    srcs = ["message_contention_bm.py"],
    main = "message_contention_bm.py",
    extensions = [
        "//gxf/std",
        "//gxf/serialization",
        "//gxf/benchmark",
    ],
    deps = [
        "//gxf/std:std_extension",
        "//gxf/benchmark:benchmark_extension"
    ],
)

nv_pygxf_app(
    name = "message_contention_1px4hx5c_6t_r_bm",
    args = ["--parallel_pipeline_size 1 --pipeline_length 4 --dummy_component_size 5 --worker_thread_number 6"],
    srcs = ["message_contention_bm.py"],
    main = "message_contention_bm.py",
    extensions = [
        "//gxf/std",
        "//gxf/serialization",
        "//gxf/benchmark",
    ],
    deps = [
        "//gxf/std:std_extension",
        "//gxf/benchmark:benchmark_extension"
    ],
)

nv_pygxf_app(
    name = "message_contention_2px4hx5c_6t_r_bm",
    args = ["--parallel_pipeline_size 2 --pipeline_length 4 --dummy_component_size 5 --worker_thread_number 6"],
    srcs = ["message_contention_bm.py"],
    main = "message_contention_bm.py",
    extensions = [
        "//gxf/std",
        "//gxf/serialization",
        "//gxf/benchmark",
    ],
    deps = [
        "//gxf/std:std_extension",
        "//gxf/benchmark:benchmark_extension"
    ],
)

nv_pygxf_app(
    name = "message_contention_4px4hx5c_6t_r_bm",
    args = ["--parallel_pipeline_size 4 --pipeline_length 4 --dummy_component_size 5 --worker_thread_number 6"],
    srcs = ["message_contention_bm.py"],
    main = "message_contention_bm.py",
    extensions = [
        "//gxf/std",
        "//gxf/serialization",
        "//gxf/benchmark",
    ],
    deps = [
        "//gxf/std:std_extension",
        "//gxf/benchmark:benchmark_extension"
    ],
)

nv_pygxf_app(
    name = "message_contention_4px4hx5c_10t_r_bm",
    args = ["--parallel_pipeline_size 4 --pipeline_length 4 --dummy_component_size 5 --worker_thread_number 10"],
    srcs = ["message_contention_bm.py"],
    main = "message_contention_bm.py",
    extensions = [
        "//gxf/std",
        "//gxf/serialization",
        "//gxf/benchmark",
    ],
    deps = [
        "//gxf/std:std_extension",
        "//gxf/benchmark:benchmark_extension"
    ],
)

nv_pygxf_app(
    name = "message_contention_4px4hx10c_6t_r_bm",
    args = ["--parallel_pipeline_size 4 --pipeline_length 4 --dummy_component_size 10 --worker_thread_number 6"],
    srcs = ["message_contention_bm.py"],
    main = "message_contention_bm.py",
    extensions = [
        "//gxf/std",
        "//gxf/serialization",
        "//gxf/benchmark",
    ],
    deps = [
        "//gxf/std:std_extension",
        "//gxf/benchmark:benchmark_extension"
    ],
)

nv_pygxf_app(
    name = "message_contention_8px4hx5c_6t_r_bm",
    args = ["--parallel_pipeline_size 8 --pipeline_length 4 --dummy_component_size 5 --worker_thread_number 6"],
    srcs = ["message_contention_bm.py"],
    main = "message_contention_bm.py",
    extensions = [
        "//gxf/std",
        "//gxf/serialization",
        "//gxf/benchmark",
    ],
    deps = [
        "//gxf/std:std_extension",
        "//gxf/benchmark:benchmark_extension"
    ],
)

nv_pygxf_app(
    name = "message_contention_8px8hx5c_6t_r_bm",
    args = ["--parallel_pipeline_size 8 --pipeline_length 8 --dummy_component_size 5 --worker_thread_number 6"],
    srcs = ["message_contention_bm.py"],
    main = "message_contention_bm.py",
    extensions = [
        "//gxf/std",
        "//gxf/serialization",
        "//gxf/benchmark",
    ],
    deps = [
        "//gxf/std:std_extension",
        "//gxf/benchmark:benchmark_extension"
    ],
)
