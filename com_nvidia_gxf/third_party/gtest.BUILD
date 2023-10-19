"""
 SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

load("@bazel_skylib//lib:selects.bzl", "selects")

cc_library(
    name = "gtest",
    srcs = glob(
        ["src/*.cc"],
        exclude = ["src/gtest-all.cc", "gtest_main.cc"],
    ),
    hdrs = glob([
        "include/**/*.h",
        "src/*.h",
    ]),
    includes = ["include"],
    linkopts =
        selects.with_or({
            "@com_nvidia_gxf//engine/build:windows": [],
            "@com_nvidia_gxf//engine/build:windows_msvc": [],
            ("@com_nvidia_gxf//engine/build:platform_hp11_sbsa",
             "@com_nvidia_gxf//engine/build:platform_hp20_sbsa",
             "@com_nvidia_gxf//engine/build:platform_hp21ea_sbsa",
             "@com_nvidia_gxf//engine/build:platform_jetpack51",
             "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_11_8",
             "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_1") : [ "-pthread"],
        }),
    visibility = ["//visibility:public"],
    linkstatic=True,
)

cc_library(
    name = "main",
    srcs = glob(
        ["src/*.cc"],
        exclude = ["src/gtest-all.cc"],
    ),
    hdrs = glob([
        "include/**/*.h",
        "src/*.h",
    ]),
    includes = ["include"],
    linkopts =
        selects.with_or({
            "@com_nvidia_gxf//engine/build:windows": [],
            "@com_nvidia_gxf//engine/build:windows_msvc": [],
            ("@com_nvidia_gxf//engine/build:platform_hp11_sbsa",
             "@com_nvidia_gxf//engine/build:platform_hp20_sbsa",
             "@com_nvidia_gxf//engine/build:platform_hp21ea_sbsa",
             "@com_nvidia_gxf//engine/build:platform_jetpack51",
             "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_11_8",
             "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_1") : [ "-pthread"],
        }),
    visibility = ["//visibility:public"],
    linkstatic=True,
)
