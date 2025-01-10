"""
Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.

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

load("@bazel_skylib//lib:selects.bzl", "selects")

####  Target platform configs
config_setting(
    name = "platform_x86_64_cuda_12_6",
    define_values = {"target_platform": "x86_64_cuda_12_6"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "platform_x86_64_cuda_12_2",
    define_values = {"target_platform": "x86_64_cuda_12_2"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "platform_x86_64_rhel9_cuda_12_2",
    define_values = {"target_platform": "x86_64_rhel9_cuda_12_2"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "platform_hp21ga_sbsa",
    define_values = {
        "target_platform": "hp21ga_sbsa",
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
    name = "platform_jetpack61",
    define_values = {"target_platform": "jetpack61"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "platform_jetpack60",
    define_values = {"target_platform": "jetpack60"},
    visibility = ["//visibility:public"],
)

#### Host OS Configs
config_setting(
    name = "host_ubuntu_22_04",
    define_values = {"os": "ubuntu_22_04"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "host_rhel9",
    define_values = {"os": "rhel9"},
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
    name = "compiler_gcc_11",
    values = {"compiler": "gcc-11"},
    visibility = ["//visibility:public"],
)

# Distro config
config_setting(
    name = "rhel",
    define_values = {"os": "rhel9"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "ubuntu",
    define_values = {"os": "ubuntu_22_04"},
    visibility = ["//visibility:public"],
)

# Alias for NVCC compilers needed for toolchain.
filegroup(
    name = "nvcc",
    srcs = select({
        "//engine/build:platform_x86_64_cuda_12_2": ["@nvcc_12_02"],
        "//engine/build:platform_x86_64_cuda_12_6": ["@nvcc_12_06"],
        "//engine/build:platform_x86_64_rhel9_cuda_12_2": ["@nvcc_12_02"],
        "//engine/build:platform_hp21ea_sbsa": ["@nvcc_12_02"],
        "//engine/build:platform_hp21ga_sbsa": ["@nvcc_12_06"],
        "//engine/build:platform_jetpack60": ["@nvcc_12_02"],
        "//engine/build:platform_jetpack61": ["@nvcc_12_06"],
        }),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "gcc_11_3_aarch64_linux_gnu_toolchain",
    srcs = ["@gcc_11_3_aarch64_linux_gnu"],
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

# Configuration settings for disabling c++11 abi
config_setting(
    name = "disable_cxx11_abi",
    define_values = {"disable_cxx11_abi": "True"},
    visibility = ["//visibility:public"],
)

# Configuration settings for using c++17 compilation flag
config_setting(
    name = "python3_10",
    define_values = {"python": "python3_10"},
    visibility = ["//visibility:public"],
)

# Config settings for choose python3
selects.config_setting_group(
    name = "rhel_python3_10",
    match_all = ["//engine/build:rhel", "//engine/build:python3_10"],
)

selects.config_setting_group(
    name = "ubuntu_python3_10",
    match_all = ["//engine/build:ubuntu", "//engine/build:python3_10"],
)

alias(
    name = "docker_base_image",
    actual = select(
        {
            "@com_nvidia_gxf//engine/build:cpu_host": "@gxf_docker_image_cuda12_2//image",
            "@com_nvidia_gxf//engine/build:cpu_aarch64": "@gxf_docker_image_aarch64_jp60//image",
            "@com_nvidia_gxf//conditions:default": "@gxf_docker_image_cuda12_2//image",
        },
        no_match_error = "Please build only for x86 or arm64 platforms only",
    ),
    tags = ["docker"],
    visibility = ["//visibility:public"],
)
