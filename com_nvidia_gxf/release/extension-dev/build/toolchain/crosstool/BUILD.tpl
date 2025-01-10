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

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "empty",
    srcs = [],
)

# This is the entry point for --crosstool_top.  Toolchains are found
# by lopping off the name of --crosstool_top and searching for
# the "${CPU}" entry in the toolchains attribute.
cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
        "local": ":cc-compiler-k8",
        "k8": ":cc-compiler-k8",
        "aarch64": ":cc-compiler-aarch64",
    },
)

cc_toolchain(
    name = "cc-compiler-k8",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":k8_toolchain_config",
    toolchain_identifier = "k8-toolchain",
)

# Android tooling requires a default toolchain for the arm64-v8a cpu.
cc_toolchain(
    name = "cc-compiler-aarch64",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":aarch64_toolchain_config",
    toolchain_identifier = "aarch64-toolchain",
)

load(":cc_toolchain_config.bzl", "cc_toolchain_config")

cc_toolchain_config(
    name = "k8_toolchain_config",
    cxx_builtin_include_directories = [
        "/usr/include",
        "/usr/include/c++/11",
        "/usr/include/c++/11/backward",
        "/usr/include/x86_64-linux-gnu",
        "/usr/include/x86_64-linux-gnu/c++/11",
        "/usr/lib/gcc/x86_64-linux-gnu/11/include",
        "/usr/local/include",
    ],
    cxx_compile_opts = [
        "-D_DEFAULT_SOURCE",
        "-U_FORTIFY_SOURCE",
        "-fstack-protector",
        "-Wall",
        "-Wunused-result",
        "-Werror",
        "-B/usr/bin",
        "-Wunused-but-set-parameter",
        "-Wno-free-nonheap-object",
        "-fno-omit-frame-pointer",
        "-fPIC",
        # Specific for glibc https://en.cppreference.com/w/cpp/types/integer
        "-D__STDC_FORMAT_MACROS",
        "-DNDEBUG",
        "-D_FORTIFY_SOURCE=2",
        "-ffunction-sections",
        "-fdata-sections",
    ],
    cxx_compiler = "/usr/bin/g++-11",
    cxx_link_opts = [
        "-lstdc++",
        "-lm",
        "-fuse-ld=gold",
        "-Wl,-no-as-needed",
        "-Wl,-z,relro,-z,now",
        "-B/usr/bin",
        "-pass-exit-codes",
        "-fPIC",
        "-Wl,--gc-sections",
        "-Wl,--disable-new-dtags",
    ],
)

cc_toolchain_config(
    name = "aarch64_toolchain_config",
    cxx_builtin_include_directories = [
        "/usr/aarch64-linux-gnu/include",
        "/usr/aarch64-linux-gnu/include/c++/11",
        "/usr/lib/gcc-cross/aarch64-linux-gnu/11/include",
        "/usr/lib/gcc-cross/aarch64-linux-gnu/11/include-fixed",
        "/usr/aarch64-linux-gnu/include/c++/11/backward",
        "/usr/aarch64-linux-gnu/include",
    ],
    cxx_compile_opts = [
        "-D_DEFAULT_SOURCE",
        "-U_FORTIFY_SOURCE",
        "-Wall",
        "-Werror",
        "-Wunused-but-set-parameter",
        "-Wno-attributes",
        "-Wno-free-nonheap-object",
        "-fno-omit-frame-pointer",
        # Specific for glibc https://en.cppreference.com/w/cpp/types/integer
        "-D__STDC_FORMAT_MACROS",
        "-fPIC",
        "-DNDEBUG",
        "-D_FORTIFY_SOURCE=2",
        "-ffunction-sections",
        "-fdata-sections",
    ],
    cxx_compiler = "/usr/bin/aarch64-linux-gnu-g++-11",
    cxx_link_opts = [
        "-lstdc++",
        "-Wl,--dynamic-linker=/lib/ld-linux-aarch64.so.1",
        "-lm",
        "-fPIC",
        "-Wl,--gc-sections",
        "-Wl,--disable-new-dtags",
    ],
)
