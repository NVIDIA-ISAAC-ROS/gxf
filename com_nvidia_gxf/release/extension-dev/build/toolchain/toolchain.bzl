"""
 SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

_GCC_PATH_MAP = {
    "x86_64": "/usr/bin/gcc",
    "jetson": "/usr/bin/aarch64-linux-gnu-gcc-11",
}

def _tpl(repository_ctx, tpl, substitutions = {}, out = None):
    repository_ctx.template(
        out,
        Label("@com_extension_dev//build/toolchain/crosstool:%s.tpl" % tpl),
        substitutions,
    )

def _toolchain_impl(repository_ctx):
    target_platform = repository_ctx.os.environ[_TARGET_PLATFORM]
    substitutions = {
        "%{gcc_path}": _GCC_PATH_MAP[target_platform],
    }
    host_substitutions = {
        "%{gcc_path}": _GCC_PATH_MAP["x86_64"],
    }
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
