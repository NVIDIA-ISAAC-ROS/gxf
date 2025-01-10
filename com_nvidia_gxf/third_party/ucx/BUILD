"""
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

cc_library(
    name = "ucx",
    deps = select({
        "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_2": ["@ucx_x86_64_cuda_12_2"],
        "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_6": ["@ucx_x86_64_cuda_12_6"],
        "@com_nvidia_gxf//engine/build:platform_x86_64_rhel9_cuda_12_2": ["@ucx_x86_64_rhel9_cuda_12_2"],
        "@com_nvidia_gxf//engine/build:platform_hp21ea_sbsa": ["@ucx_aarch64_hp21ea_sbsa"],
        "@com_nvidia_gxf//engine/build:platform_hp21ga_sbsa": ["@ucx_aarch64_hp21ga_sbsa"],
        "@com_nvidia_gxf//engine/build:platform_jetpack60": ["@ucx_aarch64_jetpack60"],
        "@com_nvidia_gxf//engine/build:platform_jetpack61": ["@ucx_aarch64_jetpack61"],
    }),
    visibility = ["//visibility:public"],
)