"""
 SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
            path = ctx.attr.cxx_ld,
        ),
        tool_path(
            name = "ar",
            path = ctx.attr.ar,
        ),
        tool_path(
            name = "cpp",
            path = ctx.attr.cxx_compiler,
        ),
        tool_path(
            name = "gcov",
            path = ctx.attr.gcov,
        ),
        tool_path(
            name = "nm",
            path = ctx.attr.nm,
        ),
        tool_path(
            name = "objdump",
            path = ctx.attr.objdump,
        ),
        tool_path(
            name = "strip",
            path = ctx.attr.strip,
        ),
    ]

    cxx14_feature = feature(
        name = "c++14",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-std=c++14"])],
            ),
        ],
    )

    cxx17_feature = feature(
        name = "c++17",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-std=c++17"])],
            ),
        ],
    )

    disable_cxx11_abi_feature = feature(
        name = "c++11_dual_abi",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [flag_group(flags = ["-D_GLIBCXX_USE_CXX11_ABI=0"])],
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

    include_directories_feature = feature(
        name = "toolchain_include_directories",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.clif_match,
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-isystem",
                            "external/gcc_9_3_aarch64_linux_gnu/aarch64-buildroot-linux-gnu/include/c++/9.3.0/",
                            "-isystem",
                            "external/gcc_9_3_aarch64_linux_gnu/aarch64-buildroot-linux-gnu/include/c++/9.3.0/aarch64-buildroot-linux-gnu/",
                            "-isystem",
                            "external/gcc_9_3_aarch64_linux_gnu/aarch64-buildroot-linux-gnu/lib",
                            "-isystem",
                            "external/gcc_9_3_aarch64_linux_gnu/aarch64-buildroot-linux-gnu/lib64",
                            "-isystem",
                            "external/gcc_9_3_aarch64_linux_gnu/aarch64-buildroot-linux-gnu/bin",
                            "-isystem",
                            "external/gcc_9_3_aarch64_linux_gnu/aarch64-buildroot-linux-gnu/sysroot/usr/include/",
                            "-isystem",
                            "external/gcc_9_3_aarch64_linux_gnu/lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/include/",
                            "-isystem",
                            "external/gcc_9_3_aarch64_linux_gnu/lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/include-fixed/",
                        ],
                    ),
                ],
            ),
        ],
    )

    features = [
           cxx_compile_opts_feature,
           opt_feature,
           dbg_feature,
           fastbuild_feature,
           linker_opts_feature,
          ]

    if ctx.attr.enable_aarch64_gcc_9_includes:
        features.append(include_directories_feature)

    if ctx.attr.enable_cxx_17_feature:
        features.append(cxx17_feature)
    else:
        features.append(cxx14_feature)

    if ctx.attr.disable_cxx11_abi_feature:
        features.append(disable_cxx11_abi_feature)

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = "gcc-toolchain",
        host_system_name = "i686-unknown-linux-gnu",
        target_system_name = "i686-unknown-linux-gnu",
        target_cpu = "k8",
        target_libc = "glibc-2.19",
        compiler = "nvcc-11.6-gcc-9.4.0",
        abi_version = "gcc-9.4.0",
        abi_libc_version = "glibc-2.19",
        tool_paths = tool_paths,
        cxx_builtin_include_directories = ctx.attr.cxx_builtin_include_directories,
        features = features,
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {
        "cxx_builtin_include_directories": attr.string_list(default = []),
        "cxx_compile_opts": attr.string_list(default = []),
        "cxx_link_opts": attr.string_list(default = []),
        "cxx_compiler": attr.string(mandatory = True),
        "cxx_ld": attr.string(default = "/usr/bin/ld"),
        "ar": attr.string(default = "/usr/bin/ar"),
        "gcov": attr.string(default = "/usr/bin/gcov"),
        "nm": attr.string(default = "/usr/bin/nm"),
        "objdump": attr.string(default = "/usr/bin/objdump"),
        "strip": attr.string(default = "/usr/bin/strip"),
        "enable_aarch64_gcc_9_includes": attr.bool(default = False),
        "enable_cxx_17_feature": attr.bool(default = False),
        "disable_cxx11_abi_feature": attr.bool(default = False),
    },
    provides = [CcToolchainConfigInfo],
)
