"""
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("@rules_cc//cc:defs.bzl", "cc_test")
load("//gxf/python:pygxf.bzl", "nv_pygxf_test")

cc_test(
    name = "test_video_format",
    srcs = ["test_video_format.cpp"],
    data = [
        "//gxf/multimedia:libgxf_multimedia.so",
        "//gxf/std:libgxf_std.so",
    ],
    deps = [
        "//common",
        "//gxf/core:gxf",
        "//gxf/multimedia",
        "//gxf/multimedia:video",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_video",
    srcs = ["test_video.cpp"],
    data = [
        "//gxf/multimedia:libgxf_multimedia.so",
        "//gxf/std:libgxf_std.so",
    ],
    deps = [
        "//common",
        "//gxf/core:gxf",
        "//gxf/multimedia",
        "//gxf/multimedia:video",
        "@dlpack",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_audio",
    srcs = ["test_audio.cpp"],
    data = [
        "//gxf/multimedia:libgxf_multimedia.so",
        "//gxf/std:libgxf_std.so",
    ],
    deps = [
        "//gxf/core:gxf",
        "//gxf/multimedia",
        "//gxf/multimedia:audio",
        "@gtest//:main",
    ],
)

nv_pygxf_test(
    name = "test_video_bindings",
    srcs = ["test_video_bindings.py"],
    app = "gxf/multimedia/tests/test_ping.yaml",
    data = [],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/cuda",
        "//gxf/python_codelet",
        "//gxf/multimedia",
    ],
    manifest_path_hack = "gxf/multimedia/tests/",
    tags = [
        "manual",
        "pytest",
    ],
    deps = [
        "//gxf/cuda:cuda_extension",
        "//gxf/python",
        "//gxf/python_codelet:python_codelet_extension",
        "//gxf/sample:sample_extension",
        "//gxf/std:std_extension",
        "//gxf/multimedia:multimedia_extension"
    ],
)
