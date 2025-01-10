"""
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

cc_library(
    name = "libnuma1_aarch64",
    srcs = [
        "usr/lib/aarch64-linux-gnu/libnuma.so.1",
        "usr/lib/aarch64-linux-gnu/libnuma.so.1.0.0",
    ],
    linkopts = [
        "-Wl,--no-as-needed," +
        "--as-needed",
    ],
    visibility = ["//visibility:public"],
)