"""
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//gxf:gxf.bzl", "nv_gxf_cc_library")

nv_gxf_cc_library(
    name = "exponential_moving_average",
    hdrs = ["exponential_moving_average.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
    ],
)

nv_gxf_cc_library(
    name = "fast_running_median",
    hdrs = ["fast_running_median.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
    ],
)

nv_gxf_cc_library(
    name = "time",
    srcs = ["time.cpp"],
    hdrs = ["time.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:expected",
        "//gxf/core:gxf_c_api",
    ],
)

nv_gxf_cc_library(
    name = "storage_size",
    srcs = ["storage_size.cpp"],
    hdrs = ["storage_size.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:expected",
        "//gxf/core:gxf_c_api",
    ],
)
