"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//gxf:gxf.bzl", "nv_gxf_cc_library")

nv_gxf_cc_library(
    name = "staging_queue",
    hdrs = ["staging_queue.hpp"],
    linkopts = [
        "-pthread",  # For <mutex>
    ],
    visibility = ["//visibility:public"],
    deps = [
        "staging_queue_iterator",
        "//common",
    ],
)

nv_gxf_cc_library(
    name = "staging_queue_iterator",
    hdrs = ["staging_queue_iterator.hpp"],
    visibility = ["//visibility:public"],
)
