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

load("//third_party:cuda.bzl", "cuda_deps")
load("//third_party:nvsci.bzl", "nvsci_deps")
load("@bazel_skylib//lib:selects.bzl", "selects")

cuda_deps()

nvsci_deps()

# Select python version based on target platform (default: python 3.8)
alias(
  name = "python_aarch64",
  actual = select({
    "//engine/build:compiler_gcc_9": "@python_aarch64_3_8",
    "//conditions:default": "@python_aarch64_3_8",
    }),
  visibility = ["//visibility:public"],
)

alias(
  name = "python_x86_64_x_y",
  actual = select({
    "//engine/build:python3_7": "@python_x86_64_3_7",
    "//conditions:default": "@python_x86_64",
    }),
  visibility = ["//visibility:public"],
)

alias(
  name = "python",
  actual = selects.with_or({
        ("//engine/build:platform_x86_64_cuda_11_8",
         "//engine/build:platform_x86_64_cuda_12_1") : "//third_party:python_x86_64_x_y",
        ("//engine/build:platform_hp11_sbsa",
         "//engine/build:platform_hp20_sbsa",
         "//engine/build:platform_hp21ea_sbsa",
         "//engine/build:platform_jetpack51"): "//third_party:python_aarch64",
        }),
  visibility = ["//visibility:public"],
)
