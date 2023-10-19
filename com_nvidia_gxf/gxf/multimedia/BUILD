"""
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load(
    "//gxf:gxf.bzl",
    "nv_gxf_cc_extension",
    "nv_gxf_cc_library",
)

nv_gxf_cc_library(
    name = "multimedia_lib",
    deps = [
        "audio",
        "camera",
        "video",
    ],
)

nv_gxf_cc_extension(
    name = "multimedia",
    srcs = ["multimedia.cpp"],
    deps = [
        "multimedia_lib",
    ],
    ext_deps = ["//gxf/std"],
)

nv_gxf_cc_library(
    name = "audio",
    srcs = ["audio.cpp"],
    hdrs = ["audio.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:expected",
        "//gxf/core:gxf_cpp_api",
        "//gxf/std:allocator",
        "//gxf/std:memory_buffer",
    ],
)

nv_gxf_cc_library(
    name = "video",
    srcs = ["video.cpp"],
    hdrs = [
        "video.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:expected",
        "//gxf/core:gxf_cpp_api",
        "//gxf/std:allocator",
        "//gxf/std:memory_buffer",
        "//gxf/std:tensor",
    ],
)

nv_gxf_cc_library(
    name = "camera",
    hdrs = [
        "camera.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/std:timestamp",
        ":video",
    ],
)

load("//registry/build:registry.bzl", "register_extension")

exports_files(["LICENSE"])

register_extension(
    name = "register_multimedia_ext",
    badges = [""],
    extension = "multimedia",
    headers = [
        "BUILD.public",
        "audio.hpp",
        "camera.hpp",
        "video.hpp",
    ],
    labels = ["gxf"],
    license_file = ":LICENSE",
    local_dependencies = ["//gxf/std:register_std_ext"],
    priority = "1",
    url = "www.nvidia.com",
    uuid = "6f2d1afc-1057-481a-9da6-a5f61fed178e",
    version = "2.3.0",
    visibility = ["//visibility:public"],
)
