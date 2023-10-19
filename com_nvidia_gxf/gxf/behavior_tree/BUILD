"""
Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.

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

nv_gxf_cc_extension(
    name = "behavior_tree",
    srcs = ["behavior_tree.cpp"],
    deps = [
        "constant_behavior",
        "entity_count_failure_repeat_controller",
        "parallel_behavior",
        "repeat_behavior",
        "selector_behavior",
        "sequence_behavior",
        "switch_behavior",
        "timer_behavior",
        "//gxf/core:gxf",
        "@yaml-cpp",
    ],
    ext_deps = ["//gxf/std"],
)

nv_gxf_cc_library(
    name = "constant_behavior",
    srcs = ["constant_behavior.cpp"],
    hdrs = ["constant_behavior.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
        "//gxf/core:gxf",
        "//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "entity_count_failure_repeat_controller",
    srcs = ["entity_count_failure_repeat_controller.cpp"],
    hdrs = ["entity_count_failure_repeat_controller.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
        "//gxf/core:gxf",
        "//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "parallel_behavior",
    srcs = ["parallel_behavior.cpp"],
    hdrs = ["parallel_behavior.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
        "//gxf/core:gxf",
        "//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "repeat_behavior",
    srcs = ["repeat_behavior.cpp"],
    hdrs = ["repeat_behavior.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
        "//gxf/core:gxf",
        "//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "selector_behavior",
    srcs = ["selector_behavior.cpp"],
    hdrs = ["selector_behavior.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
        "//gxf/core:gxf",
        "//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "sequence_behavior",
    srcs = ["sequence_behavior.cpp"],
    hdrs = ["sequence_behavior.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
        "//gxf/core:gxf",
        "//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "switch_behavior",
    srcs = ["switch_behavior.cpp"],
    hdrs = ["switch_behavior.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
        "//gxf/core:gxf",
        "//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "timer_behavior",
    srcs = ["timer_behavior.cpp"],
    hdrs = ["timer_behavior.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
        "//gxf/core:gxf",
        "//gxf/std",
    ],
)

load("//registry/build:registry.bzl", "register_extension")

exports_files(["LICENSE"])

register_extension(
    name = "register_behavior_tree_ext",
    badges = [""],
    extension = "behavior_tree",
    headers = [
        "BUILD.public",
        "constant_behavior.hpp",
        "entity_count_failure_repeat_controller.hpp",
        "parallel_behavior.hpp",
        "repeat_behavior.hpp",
        "selector_behavior.hpp",
        "sequence_behavior.hpp",
        "switch_behavior.hpp",
        "timer_behavior.hpp",
    ],
    labels = ["gxf"],
    license_file = ":LICENSE",
    local_dependencies = ["//gxf/std:register_std_ext"],
    priority = "1",
    url = "www.nvidia.com",
    uuid = "9e6e55d4-1bc9-11ec-9621-0242ac130002",
    version = "0.3.0",
    visibility = ["//visibility:public"],
)
