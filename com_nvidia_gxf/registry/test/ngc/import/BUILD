"""
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load(
    "//gxf:gxf.bzl",
    "nv_gxf_cc_extension",
    "nv_gxf_app",
    "nv_gxf_ext_deps",
)

# Sample usage to demonstrate NGC imports of extensions
# Uncomment the below code block for testing locally
# Commenting this to prevent CI from trying to import from NGC

'''
# Define extension transitive dependencies
nv_gxf_ext_deps(name = "std_ext_deps",
                 ext = "libgxf_std.so",
                 ext_deps = [],
                 visibility = ["//visibility:public"])

nv_gxf_ext_deps(name = "test_ext_deps",
                 ext = "libgxf_test.so",
                 ext_deps = ["//registry/test/ngc/import:std_ext_deps"],
                 visibility = ["//visibility:public"])

nv_gxf_ext_deps(name = "sample_ext_deps",
                 ext = "libgxf_sample.so",
                 ext_deps = ["//registry/test/ngc/import:std_ext_deps"],
                 visibility = ["//visibility:public"])

alias(
    name = "libgxf_std.so",
    actual = select({

        "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_2": "@std_x86_64_cuda_12_2//:standard_extension",
        "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_6": "@std_x86_64_cuda_12_6//:standard_extension",
        "@com_nvidia_gxf//engine/build:platform_x86_64_rhel9_cuda_12_2": "@std_x86_64_rhel9_cuda_12_2//:standard_extension",
        "@com_nvidia_gxf//engine/build:cpu_aarch64": "@std_aarch64//:standard_extension",
        "@com_nvidia_gxf//engine/build:platform_hp21ea_sbsa": "@std_hp21ea_sbsa//:standard_extension",
        "@com_nvidia_gxf//engine/build:platform_hp21ga_sbsa": "@std_hp21ga_sbsa//:standard_extension",
    }),
    visibility = ["//visibility:public"],
)

alias(
    name = "libgxf_test.so",
    actual = select({
        "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_2": "@test_x86_64_cuda_12_2//:test_extension",
        "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_6": "@test_x86_64_cuda_12_6//:test_extension",
        "@com_nvidia_gxf//engine/build:platform_x86_64_rhel9_cuda_12_2": "@test_x86_64_rhel9_cuda_12_2//:test_extension",
        "@com_nvidia_gxf//engine/build:cpu_aarch64": "@test_aarch64//:test_extension",
        "@com_nvidia_gxf//engine/build:platform_hp21ea_sbsa": "@test_hp21ea_sbsa//:test_extension",
        "@com_nvidia_gxf//engine/build:platform_hp21ga_sbsa": "@test_hp21ga_sbsa//:test_extension",
    }),
    visibility = ["//visibility:public"],
)

alias(
    name = "libgxf_sample.so",
    actual = select({
        "@com_nvidia_gxf//engine/build:cpu_host": "@sample_x86_64//:sample_extension",
        "@com_nvidia_gxf//engine/build:cpu_aarch64": "@sample_aarch64//:sample_extension",
        "@com_nvidia_gxf//engine/build:cpu_aarch64_sbsa": "@sample_sbsa//:sample_extension",
    }),
    visibility = ["//visibility:public"],
)

cc_import(
    name = "std",
    shared_library = ":libgxf_std.so",
    tags = ["manual"],
    visibility = ["//visibility:public"],
)


cc_import(
    name = "test",
    shared_library = ":libgxf_test.so",
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

cc_import(
    name = "sample",
    shared_library = ":libgxf_sample.so",
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

nv_gxf_app(
    name = "test_ping",
    srcs = ["test_ping_composer.yaml"],
    extensions = [
        "//registry/test/ngc/import:std",
        "//registry/test/ngc/import:sample",
        "//registry/test/ngc/import:test",
    ],
)

'''
