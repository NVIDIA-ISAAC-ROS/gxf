"""
Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load(
    "//gxf:gxf.bzl",
    "nv_gxf_cc_extension",
)

nv_gxf_cc_extension(
    name = "test_extension",
    srcs = ["test_extension.cpp"],
    deps = [
       "//common",
       "//gxf/std:extension_factory_helper",
       "//gxf/core:gxf",
    ],
    ext_deps = ["//gxf/std"],
)
