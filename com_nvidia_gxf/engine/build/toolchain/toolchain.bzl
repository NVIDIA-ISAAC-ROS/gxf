"""
 SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    "x86_64_cuda_11_8|gcc-9": "/usr/bin/gcc-9",
    "x86_64_cuda_12_1|gcc-9": "/usr/bin/gcc-9",
    "hp11_sbsa|gcc-9": "external/gcc_9_3_aarch64_linux_gnu/bin/aarch64-linux-gcc-9.3.0",
    "hp20_sbsa|gcc-9": "external/gcc_9_3_aarch64_linux_gnu/bin/aarch64-linux-gcc-9.3.0",
    "hp21ea_sbsa|gcc-9": "external/gcc_9_3_aarch64_linux_gnu/bin/aarch64-linux-gcc-9.3.0",
    "jetpack51|gcc-9": "external/gcc_9_3_aarch64_linux_gnu/bin/aarch64-linux-gcc-9.3.0",
}

_NVCC_PATH_MAP = {
    "x86_64_cuda_11_8": "external/nvcc_11_08/bin/nvcc",
    "x86_64_cuda_12_1": "external/nvcc_12_01/bin/nvcc",
    "hp11_sbsa": "external/nvcc_11_06/bin/nvcc",
    "hp20_sbsa": "external/nvcc_11_08/bin/nvcc",
    "hp21ea_sbsa": "external/nvcc_12_01/bin/nvcc",
    "jetpack51": "external/nvcc_11_04/bin/nvcc",
}

_CUDA_CAPABILITIES_MAP = {
    "x86_64_cuda_11_8": '"5.2", "5.3",  "6.0", "6.1", "6.2", "7.0", "7.5", "8.6", "8.9", "9.0"',
    "x86_64_cuda_12_1": '"5.2", "5.3",  "6.0", "6.1", "6.2", "7.0", "7.5", "8.6", "8.9", "9.0"',
    "hp11_sbsa": '"5.2", "5.3",  "6.0", "6.1", "6.2", "7.0", "7.5", "8.6"',
    "hp20_sbsa": '"5.2", "5.3",  "6.0", "6.1", "6.2", "7.0", "7.5", "8.6", "8.9", "9.0"',
    "hp21ea_sbsa": '"5.2", "5.3",  "6.0", "6.1", "6.2", "7.0", "7.5", "8.6", "8.9", "9.0"',
    "jetpack51": '"5.3","6.2","7.2","7.5","8.6","8.7"',
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
        host_substitutions = {
            "%{gcc_path}": _GCC_PATH_MAP[target_platform + "|" + compiler],
            "%{nvcc_path}": _NVCC_PATH_MAP[target_platform],
            "%{cuda_capabilities}": _CUDA_CAPABILITIES_MAP[target_platform],
        }
    else:
        host_substitutions = {
            "%{gcc_path}": _GCC_PATH_MAP["x86_64_cuda_12_1" + "|" + compiler],
            "%{nvcc_path}": _NVCC_PATH_MAP["x86_64_cuda_12_1"],
            "%{cuda_capabilities}": _CUDA_CAPABILITIES_MAP["x86_64_cuda_12_1"],
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
