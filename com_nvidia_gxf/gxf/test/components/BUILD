"""
 SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    "nv_gxf_cc_library",
)

nv_gxf_cc_library(
    name = "components",
    visibility = ["//visibility:public"],
    deps = [
        "entity_monitor",
        "mock_allocator",
        "mock_codelet",
        "mock_failure",
        "mock_receiver",
        "mock_transmitter",
        "tensor_comparator",
        "tensor_generator",
    ],
)

nv_gxf_cc_library(
    name = "entity_monitor",
    srcs = ["entity_monitor.cpp"],
    hdrs = ["entity_monitor.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
        "//gxf/core:gxf",
        "//gxf/std",
        "@yaml-cpp",
    ],
)

nv_gxf_cc_library(
    name = "mock_allocator",
    srcs = ["mock_allocator.cpp"],
    hdrs = ["mock_allocator.hpp"],
    deps = [
        "//common",
        "//gxf/core:gxf",
        "//gxf/std",
        "//third_party:cudart",
        "@yaml-cpp",
    ],
)

nv_gxf_cc_library(
    name = "mock_receiver",
    srcs = ["mock_receiver.cpp"],
    hdrs = ["mock_receiver.hpp"],
    deps = [
        "//common",
        "//gxf/core:gxf",
        "//gxf/std",
        "@yaml-cpp",
    ],
)

nv_gxf_cc_library(
    name = "mock_transmitter",
    srcs = ["mock_transmitter.cpp"],
    hdrs = ["mock_transmitter.hpp"],
    deps = [
        "//common",
        "//gxf/core:gxf",
        "//gxf/std",
        "@yaml-cpp",
    ],
)

nv_gxf_cc_library(
    name = "tensor_comparator",
    srcs = ["tensor_comparator.cpp"],
    hdrs = ["tensor_comparator.hpp"],
    deps = [
        "//common",
        "//gxf/core:gxf",
        "//gxf/std",
        "//gxf/std:tensor",
        "//third_party:cudart",
        "@yaml-cpp",
    ],
)

nv_gxf_cc_library(
    name = "tensor_generator",
    srcs = ["tensor_generator.cpp"],
    hdrs = ["tensor_generator.hpp"],
    deps = [
        "//common",
        "//gxf/core:gxf",
        "//gxf/std",
        "//third_party:cudart",
        "@yaml-cpp",
    ],
)

nv_gxf_cc_library(
    name = "mock_failure",
    srcs = ["mock_failure.cpp"],
    hdrs = ["mock_failure.hpp"],
    deps = [
        "//common",
        "//gxf/core:gxf",
        "//gxf/std",
        "@yaml-cpp",
    ],
)

nv_gxf_cc_library(
    name = "mock_codelet",
    srcs = ["mock_codelet.cpp"],
    hdrs = ["mock_codelet.hpp"],
    deps = [
        "//common",
        "//gxf/core:gxf",
        "//gxf/std",
        "@yaml-cpp",
    ],
)
