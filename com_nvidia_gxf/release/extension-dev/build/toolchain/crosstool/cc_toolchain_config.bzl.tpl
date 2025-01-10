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

load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "action_config",
    "env_entry",
    "env_set",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
    "tool",
    "tool_path",
    "variable_with_value",
)
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")

def _impl(ctx):
    tool_paths = [
        tool_path(
            name = "gcc",
            path = ctx.attr.cxx_compiler,
        ),
        tool_path(
            name = "ld",
            path = "/usr/bin/ld",
        ),
        tool_path(
            name = "ar",
            path = "/usr/bin/ar",
        ),
        tool_path(
            name = "cpp",
            path = ctx.attr.cxx_compiler,
        ),
        tool_path(
            name = "gcov",
            path = "/usr/bin/gcov",
        ),
        tool_path(
            name = "nm",
            path = "/usr/bin/nm",
        ),
        tool_path(
            name = "objdump",
            path = "/usr/bin/objdump",
        ),
        tool_path(
            name = "strip",
            path = "/usr/bin/strip",
        ),
    ]

    cpp17_feature = feature(
        name = "c++17",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-std=c++17"])],
            ),
        ],
    )

    cxx_compile_opts_feature = feature(
        name = "c++_compile_opts",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ctx.attr.cxx_compile_opts,
                    ),
                ],
            ),
        ],
    )

    dbg_feature = feature(
        name = "dbg",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-Og",
                            "-ggdb3",
                        ],
                    ),
                ],
            ),
        ],
        implies = [],
    )

    fastbuild_feature = feature(
        name = "fastbuild",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-O0",
                            "-g0",
                        ],
                    ),
                ],
            ),
        ],
        implies = [],
    )

    opt_feature = feature(
        name = "opt",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-O3",
                            "-ggdb2",
                        ],
                    ),
                ],
            ),
        ],
        implies = [],
    )

    linker_opts_feature = feature(
        name = "linker_opts",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [
                    flag_group(
                        flags = ctx.attr.cxx_link_opts,
                    ),
                ],
            ),
        ],
    )

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = "gcc-toolchain",
        host_system_name = "i686-unknown-linux-gnu",
        target_system_name = "i686-unknown-linux-gnu",
        target_cpu = "k8",
        target_libc = "glibc-2.19",
        compiler = "nvcc-11.6-gcc-9.3.0",
        abi_version = "gcc-11.3.0",
        abi_libc_version = "glibc-2.19",
        tool_paths = tool_paths,
        cxx_builtin_include_directories = ctx.attr.cxx_builtin_include_directories,
        features = [
            cpp17_feature,
            cxx_compile_opts_feature,
            opt_feature,
            dbg_feature,
            fastbuild_feature,
            linker_opts_feature,
        ],
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {
        "cxx_builtin_include_directories": attr.string_list(default = []),
        "cxx_compile_opts": attr.string_list(default = []),
        "cxx_link_opts": attr.string_list(default = []),
        "cxx_compiler": attr.string(),
    },
    provides = [CcToolchainConfigInfo],
)
