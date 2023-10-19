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
    name = "first_fit_allocator_base",
    srcs = ["first_fit_allocator_base.cpp"],
    hdrs = ["first_fit_allocator_base.hpp"],
    visibility = ["//visibility:public"],
    deps = ["//common:expected"],
)

nv_gxf_cc_library(
    name = "first_fit_allocator",
    hdrs = ["first_fit_allocator.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        ":first_fit_allocator_base",
        "//common:expected",
    ],
)
