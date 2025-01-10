"""
 SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

load("//third_party:cuda.bzl", "cuda_deps")
load("//third_party:nvsci.bzl", "nvsci_deps")
load("@bazel_skylib//lib:selects.bzl", "selects")

cuda_deps()

nvsci_deps()

# Select python version based on target platform (default: python 3.10)
alias(
  name = "python_aarch64",
  actual = select({
    "//engine/build:compiler_gcc_11": "@python_aarch64_3_10",
    "//conditions:default": "@python_aarch64_3_10",
    }),
  visibility = ["//visibility:public"],
)

alias(
  name = "python_x86_64",
  actual = select({
    "//engine/build:ubuntu_python3_10": "@python_x86_64_3_10",
    "//engine/build:rhel_python3_10": "@python_x86_64_rhel9_3_10",
    "//conditions:default": "@python_x86_64_3_10",
    }),
  visibility = ["//visibility:public"],
)

alias(
  name = "python",
  actual = selects.with_or({
        ("//engine/build:platform_x86_64_cuda_12_2",
         "//engine/build:platform_x86_64_cuda_12_6",
         "//engine/build:platform_x86_64_rhel9_cuda_12_2") : "//third_party:python_x86_64",
        ("//engine/build:platform_hp21ea_sbsa",
         "//engine/build:platform_hp21ga_sbsa",
         "//engine/build:platform_jetpack60",
         "//engine/build:platform_jetpack61"): "//third_party:python_aarch64",
        }),
  visibility = ["//visibility:public"],
)
