"""
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load(
    "//gxf:gxf.bzl",
    "nv_gxf_cc_library",
    "nv_gxf_cc_binary"
)

load("@com_nvidia_gxf//packaging:packaging.bzl", "nv_gxf_pkg_library")

nv_gxf_cc_library(
    name = "common_logger",
    srcs = ["logger.cpp"],
    hdrs = ["logger.hpp"],
    visibility = ["//visibility:public"],
)

nv_gxf_cc_library(
    name = "gxf_logger",
    srcs = ["gxf_logger.cpp"],
    hdrs = ["gxf_logger.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//common:singleton",
        "common_logger",
    ],
)

nv_gxf_cc_binary(
    name = "libgxf_logger.so",
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [
        "common_logger",
    ],
    linkopts = [
        "-Wl,-soname,libgxf_logger.so",
    ],
)

cc_import(
    name = "libgxf_logger",
    hdrs = [
        "logger.hpp",
    ],
    shared_library = "libgxf_logger.so",
    visibility = ["//visibility:public"],
)

nv_gxf_pkg_library(
    name = "logger",
    build = "BUILD.release",
    headers = [
        "logger.hpp",
    ],
    binaries = [
        ":libgxf_logger.so",
    ]
)
