"""
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load(
    "//gxf:gxf.bzl",
    "nv_gxf_cc_library",
)

nv_gxf_cc_library(
    name = "camera_message",
    srcs = ["camera_message.cpp"],
    hdrs = ["camera_message.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/std/gems/video_buffer",
        "//gxf/multimedia",
        "//gxf/std",
    ],
)