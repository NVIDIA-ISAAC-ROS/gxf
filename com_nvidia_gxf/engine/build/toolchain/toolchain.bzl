"""
 SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

_TARGET_PLATFORM = "target_platform"
_COMPLIER = "compiler"
_CPU = "cpu"

_GCC_PATH_MAP = {
    "x86_64_cuda_12_6|gcc-11": "/usr/bin/gcc-11",
    "x86_64_cuda_12_2|gcc-11": "/usr/bin/gcc-11",
    "x86_64_rhel9_cuda_12_2|gcc-11": "/usr/bin/gcc-11",
    "hp21ea_sbsa|gcc-11": "external/gcc_11_3_aarch64_linux_gnu/bin/aarch64-linux-gcc-11.3.0",
    "hp21ga_sbsa|gcc-11": "external/gcc_11_3_aarch64_linux_gnu/bin/aarch64-linux-gcc-11.3.0",
    "jetpack60|gcc-11": "external/gcc_11_3_aarch64_linux_gnu/bin/aarch64-linux-gcc-11.3.0",
    "jetpack61|gcc-11": "external/gcc_11_3_aarch64_linux_gnu/bin/aarch64-linux-gcc-11.3.0",
}

_NVCC_PATH_MAP = {
    "x86_64_cuda_12_6": "external/nvcc_12_06/bin/nvcc",
    "x86_64_cuda_12_2": "external/nvcc_12_02/bin/nvcc",
    "x86_64_rhel9_cuda_12_2": "external/nvcc_12_02/bin/nvcc",
    "hp21ea_sbsa": "external/nvcc_12_02/bin/nvcc",
    "hp21ga_sbsa": "external/nvcc_12_06/bin/nvcc",
    "jetpack60": "external/nvcc_12_02/bin/nvcc",
    "jetpack61": "external/nvcc_12_06/bin/nvcc",
}

_CUDA_CAPABILITIES_MAP = {
    "x86_64_cuda_12_6": '"5.2", "5.3",  "6.0", "6.1", "6.2", "7.0", "7.5", "8.6", "8.9", "9.0"',
    "x86_64_cuda_12_2": '"5.2", "5.3",  "6.0", "6.1", "6.2", "7.0", "7.5", "8.6", "8.9", "9.0"',
    "x86_64_rhel9_cuda_12_2": '"5.2", "5.3",  "6.0", "6.1", "6.2", "7.0", "7.5", "8.6", "8.9", "9.0"',
    "hp21ea_sbsa": '"5.2", "5.3",  "6.0", "6.1", "6.2", "7.0", "7.5", "8.6", "8.9", "9.0"',
    "hp21ga_sbsa": '"5.2", "5.3",  "6.0", "6.1", "6.2", "7.0", "7.5", "8.6", "8.9", "9.0"',
    "jetpack60": '"5.3","6.2","7.2","7.5","8.6","8.7"',
    "jetpack61": '"5.3","6.2","7.2","7.5","8.6","8.7"',
}

def _tpl(repository_ctx, tpl, substitutions = {}, out = None):
    repository_ctx.template(
        out,
        Label("//engine/build/toolchain/crosstool:%s.tpl" % tpl),
        substitutions,
    )

def _toolchain_impl(repository_ctx):
    target_platform = repository_ctx.os.environ[_TARGET_PLATFORM]
    cpu = repository_ctx.os.environ[_CPU]
    compiler = repository_ctx.os.environ[_COMPLIER]

    substitutions = {
        "%{gcc_path}": _GCC_PATH_MAP[target_platform + "|" + compiler],
        "%{nvcc_path}": _NVCC_PATH_MAP[target_platform],
        "%{cuda_capabilities}": _CUDA_CAPABILITIES_MAP[target_platform],
    }

    if cpu == "k8":
        host_substitutions = substitutions
    else:
        if target_platform == "jetpack61" or target_platform == "hp21ga_sbsa":
            host_config = "x86_64_cuda_12_6"
        else:
            host_config = "x86_64_cuda_12_2"
        host_substitutions = {
            "%{gcc_path}": _GCC_PATH_MAP[host_config + "|" + compiler],
            "%{nvcc_path}": _NVCC_PATH_MAP[host_config],
            "%{cuda_capabilities}": _CUDA_CAPABILITIES_MAP[host_config],
        }

    _tpl(
        repository_ctx,
        "crosstool_wrapper_driver_is_not_gcc.py",
        host_substitutions,
        "crosstool/scripts/crosstool_wrapper_driver_is_not_gcc_host.py",
    )
    _tpl(
        repository_ctx,
        "crosstool_wrapper_driver_is_not_gcc.py",
        substitutions,
        "crosstool/scripts/crosstool_wrapper_driver_is_not_gcc.py",
    )
    _tpl(
        repository_ctx,
        "BUILD",
        substitutions,
        "crosstool/BUILD",
    )
    _tpl(
        repository_ctx,
        "cc_toolchain_config.bzl",
        substitutions,
        "crosstool/cc_toolchain_config.bzl",
    )

toolchain_configure = repository_rule(
    environ = [
        _TARGET_PLATFORM,
    ],
    implementation = _toolchain_impl,
)
