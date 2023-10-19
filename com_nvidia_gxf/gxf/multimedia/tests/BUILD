"""
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

cc_test(
    name = "test_video_format",
    srcs = ["test_video_format.cpp"],
    data = [
        "//gxf/multimedia:libgxf_multimedia.so",
        "//gxf/std:libgxf_std.so",
    ],
    deps = [
        "//common",
        "//gxf/core:expected",
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
        "//gxf/core:expected",
        "//gxf/multimedia",
        "//gxf/multimedia:video",
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
        "//gxf/core:expected",
        "//gxf/multimedia",
        "//gxf/multimedia:audio",
        "@gtest//:main",
    ],
)
