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
        "k8|gcc-9": ":cc-compiler-k8-gcc-9",
        "aarch64|gcc-9": ":cc-compiler-aarch64-gcc-9",
        "aarch64_sbsa|gcc-9": ":cc-compiler-aarch64-gcc-9",
    },
)

cc_toolchain(
    name = "cc-compiler-k8-gcc-9",
    all_files = ":gcc_or_nvcc",
    compiler_files = ":gcc_or_nvcc",
    dwp_files = ":empty",
    linker_files = ":gcc_or_nvcc",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":k8_toolchain_config_gcc_9",
    toolchain_identifier = "k8-toolchain-gcc-9",
)

# aarch64 toolchain
cc_toolchain(
    name = "cc-compiler-aarch64-gcc-9",
    all_files = ":gcc_or_nvcc",
    compiler_files = ":gcc_or_nvcc",
    dwp_files = ":empty",
    linker_files = ":gcc_or_nvcc",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":aarch64_toolchain_config_gcc_9",
    toolchain_identifier = "aarch64-toolchain-gcc-9",
)

filegroup(
    name = "gcc_or_nvcc",
    srcs = [
        "scripts/crosstool_wrapper_driver_is_not_gcc.py",
        "scripts/crosstool_wrapper_driver_is_not_gcc_host.py",
        "@com_nvidia_gxf//engine/build:nvcc",
    ] + select({
        "@com_nvidia_gxf//engine/build:platform_hp11_sbsa":
                ["@com_nvidia_gxf//engine/build:gcc_9_3_aarch64_linux_gnu_toolchain"],
        "@com_nvidia_gxf//engine/build:platform_hp20_sbsa":
                ["@com_nvidia_gxf//engine/build:gcc_9_3_aarch64_linux_gnu_toolchain"],
        "@com_nvidia_gxf//engine/build:platform_hp21ea_sbsa":
                ["@com_nvidia_gxf//engine/build:gcc_9_3_aarch64_linux_gnu_toolchain"],
        "@com_nvidia_gxf//engine/build:platform_jetpack51":
                ["@com_nvidia_gxf//engine/build:gcc_9_3_aarch64_linux_gnu_toolchain"],
        "//conditions:default": [],
    })
)

load(":cc_toolchain_config.bzl", "cc_toolchain_config")

cc_toolchain_config(
    name = "k8_toolchain_config_gcc_9",
    cxx_builtin_include_directories = [
        "/usr/include",
        "/usr/include/c++/9",
        "/usr/include/c++/9/backward",
        "/usr/include/x86_64-linux-gnu",
        "/usr/include/x86_64-linux-gnu/c++/9",
        "/usr/lib/gcc/x86_64-linux-gnu/9/include-fixed",
        "/usr/lib/gcc/x86_64-linux-gnu/9/include",
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
    cxx_compiler = "scripts/crosstool_wrapper_driver_is_not_gcc_host.py",
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
    enable_cxx_17_feature = select({
                            "@com_nvidia_gxf//engine/build:cxx_17": True,
                            "//conditions:default": False,
                            }),
    disable_cxx11_abi_feature = select({
                                    "@com_nvidia_gxf//engine/build:disable_cxx11_abi": True,
                                    "//conditions:default": False,
                                   })
)

cc_toolchain_config(
    name = "aarch64_toolchain_config_gcc_9",
    cxx_builtin_include_directories = [
        "external/gcc_9_3_aarch64_linux_gnu/aarch64-buildroot-linux-gnu/include/c++/9.3.0/",
        "external/gcc_9_3_aarch64_linux_gnu/aarch64-buildroot-linux-gnu/sysroot/usr/include/",
        "external/gcc_9_3_aarch64_linux_gnu/lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/include/",
        "external/gcc_9_3_aarch64_linux_gnu/lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/include-fixed/",
        "external/gcc_9_3_aarch64_linux_gnu/aarch64-buildroot-linux-gnu/include/c++/9.3.0/aarch64-buildroot-linux-gnu/",
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
        "--sysroot=external/gcc_9_3_aarch64_linux_gnu/aarch64-buildroot-linux-gnu/sysroot",
    ],
    cxx_compiler = "scripts/crosstool_wrapper_driver_is_not_gcc.py",
    cxx_link_opts = [
        "-lstdc++",
        "-lc",
        "-Wl,--dynamic-linker=/lib/ld-linux-aarch64.so.1",
        "-lm",
        "-fPIC",
        "-Wl,--gc-sections",
        "-Wl,--disable-new-dtags",
        "--sysroot=external/gcc_9_3_aarch64_linux_gnu/aarch64-buildroot-linux-gnu/sysroot",
        "-L external/gcc_9_3_aarch64_linux_gnu/aarch64-buildroot-linux-gnu/sysroot/usr/",
        "-L external/gcc_9_3_aarch64_linux_gnu/aarch64-buildroot-linux-gnu/sysroot/usr/lib/",
        "-L external/gcc_9_3_aarch64_linux_gnu/aarch64-buildroot-linux-gnu/sysroot/",
        "-L external/gcc_9_3_aarch64_linux_gnu/aarch64-buildroot-linux-gnu/sysroot/lib/",
        "-L external/gcc_9_3_aarch64_linux_gnu/aarch64-buildroot-linux-gnu/lib64/",
    ],
    cxx_ld = "external/gcc_9_3_aarch64_linux_gnu/aarch64-buildroot-linux-gnu/bin/ld",
    #ar = "external/gcc_9_3_aarch64_linux_gnu/bin/aarch64-linux-ar",
    nm = "external/gcc_9_3_aarch64_linux_gnu/bin/aarch64-linux-nm",
    gcov = "external/gcc_9_3_aarch64_linux_gnu/bin/aarch64-linux-gcov",
    objdump = "external/gcc_9_3_aarch64_linux_gnu/bin/aarch64-linux-objdump",
    strip = "external/gcc_9_3_aarch64_linux_gnu/bin/aarch64-linux-strip",
    enable_aarch64_gcc_9_includes = True,
    enable_cxx_17_feature = select({
                                    "@com_nvidia_gxf//engine/build:cxx_17": True,
                                    "//conditions:default": False,
                                   }),
    disable_cxx11_abi_feature = select({
                                    "@com_nvidia_gxf//engine/build:disable_cxx11_abi": True,
                                    "//conditions:default": False,
                                   })
)
