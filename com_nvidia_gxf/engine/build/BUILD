"""
Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.

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

# Configuration conditions to detect the target platform being built for.
# Host is always assumed to be x86_64. Both Jetson TX2 and Jetson Xavier imply AArch64.
# For more detail refer to
# https://docs.bazel.build/versions/master/configurable-attributes.html

####  Target platform configs
config_setting(
    name = "platform_x86_64_cuda_11_8",
    define_values = {"target_platform": "x86_64_cuda_11_8"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "platform_x86_64_cuda_12_1",
    define_values = {"target_platform": "x86_64_cuda_12_1"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "platform_hp11_sbsa",
    define_values = {
        "target_platform": "hp11_sbsa",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "platform_hp20_sbsa",
    define_values = {
        "target_platform": "hp20_sbsa",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "platform_hp21ea_sbsa",
    define_values = {
        "target_platform": "hp21ea_sbsa",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "platform_jetpack51",
    define_values = {"target_platform": "jetpack51"},
    visibility = ["//visibility:public"],
)

#### CPU Configs
config_setting(
    name = "cpu_host",
    values = {"cpu": "k8"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "cpu_aarch64",
    values = {"cpu": "aarch64"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "cpu_aarch64_sbsa",
    values = {"cpu": "aarch64_sbsa"},
    visibility = ["//visibility:public"],
)

#### Compiler Configs
config_setting(
    name = "compiler_gcc_9",
    values = {"compiler": "gcc-9"},
    visibility = ["//visibility:public"],
)

# Alias for NVCC compilers needed for toolchain.
filegroup(
    name = "nvcc",
    srcs = select({
        "//engine/build:platform_x86_64_cuda_11_8": ["@nvcc_11_08"],
        "//engine/build:platform_x86_64_cuda_12_1": ["@nvcc_12_01"],
        "//engine/build:platform_hp11_sbsa": ["@nvcc_11_06"],
        "//engine/build:platform_hp20_sbsa": ["@nvcc_11_08"],
        "//engine/build:platform_hp21ea_sbsa": ["@nvcc_12_01"],
        "//engine/build:platform_jetpack51": ["@nvcc_11_04"],
        }),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "gcc_9_3_aarch64_linux_gnu_toolchain",
    srcs = ["@gcc_9_3_aarch64_linux_gnu"],
    visibility = ["//visibility:public"],
)

# Configuration setting to detect if we are building without GPU support
config_setting(
    name = "isaac_gpu_disabled",
    define_values = {"gpu": "0"},
    visibility = ["//visibility:public"],
)

# Configuration settings for windows
config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)

# Configuration settings for windows when using msvc compiler
config_setting(
    name = "windows_msvc",
    values = {"cpu": "x64_windows_msvc"},
    visibility = ["//visibility:public"],
)

# Configuration settings for using c++17 compilation flag
config_setting(
    name = "cxx_17",
    define_values = {"cpp_standard": "gcc_cxx_17"},
    visibility = ["//visibility:public"],
)

# Configuration settings for disabling c++11 abi
config_setting(
    name = "disable_cxx11_abi",
    define_values = {"disable_cxx11_abi": "True"},
    visibility = ["//visibility:public"],
)

# Configuration settings for using c++17 compilation flag
config_setting(
    name = "python3_7",
    define_values = {"python": "python3_7"},
    visibility = ["//visibility:public"],
)
