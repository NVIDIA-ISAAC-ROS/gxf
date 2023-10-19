"""
Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("@bazel_skylib//lib:selects.bzl", "selects")
load("//gxf:pybind.bzl", "pybind_library")

config_setting(
    name = "enable_pygxf",
    define_values = {"dummy_pygxf": "false"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "disable_pygxf",
    define_values = {"dummy_pygxf": "true"},
    visibility = ["//visibility:public"],
)

alias(
    name = "select_pygxf",
    actual = selects.with_or({
        # pygxf bindings is used by registry cli to register cross platform targets on x86 hosts
        # Use dummy targets to avoid loading cross platform binaries on x86
        ("disable_pygxf",
         "//engine/build:platform_hp11_sbsa",
         "//engine/build:platform_hp20_sbsa",
         "//engine/build:platform_hp21ea_sbsa",
         "//engine/build:platform_jetpack51") : "pygxf_dummy",
        # pygxf is enabled for all x86 builds
        "//conditions:default" : "pygxf"
    }),
     visibility = ["//visibility:public"],
)

py_library (
    name = "pygxf_dummy",
    srcs = ["pygxf.py"],
    visibility = ["//visibility:public"],
)

pybind_library (
    name = "pygxf",
    cc_srcs = glob([
        "*.cpp",
        "*.hpp",
    ]),
    cc_deps = [
        "@com_nvidia_gxf//gxf/core:gxf",
        "@com_nvidia_gxf//common",
    ],
)
