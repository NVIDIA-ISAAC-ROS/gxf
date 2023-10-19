"""
 SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

nv_gxf_cc_extension(
    name = "test",
    srcs = ["test.cpp"],
    deps = [
        "test_helpers",
        "test_parameters",
        "test_metric",
        "//common:common",
        "//gxf/std:default_extension",
        "//gxf/std:metric",
        "//gxf/std:yaml_file_loader",
        "//gxf/test/components",
        "@yaml-cpp",
    ],
    ext_deps = [
        "//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "test_helpers",
    hdrs = ["test_helpers.hpp"],
    deps = [
    ],
)

nv_gxf_cc_library(
    name = "test_parameters",
    hdrs = ["test_parameters.hpp"],
    deps = [
    ],
)

nv_gxf_cc_library(
    name = "test_metric",
    hdrs = ["test_metric.hpp"],
    deps = [
        "@gtest//:main",
    ],
)

exports_files(["LICENSE"])
