"""
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.

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

nv_gxf_cc_binary(
    name = "libgxf_app.so",
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [
            "app",
            "arg",
            "arg_parse",
            "config_parser",
            "entity_group",
            "extension_manager",
            "graph_entity",
            "graph_utils",
            "proxy_component",
            "segment",
            ],
    linkopts = [
        "-Wl,-soname,libgxf_app.so",
    ],
)

nv_gxf_cc_library(
    name = "config_parser",
    hdrs = ["config_parser.hpp"],
    srcs = ["config_parser.cpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/std",
        "@yaml-cpp",
    ],
)

nv_gxf_cc_library(
    name = "arg",
    hdrs = ["arg.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf",
        "//gxf/std",
        "@yaml-cpp",
    ],
)

nv_gxf_cc_library(
    name = "arg_parse",
    hdrs = ["arg_parse.hpp"],
    srcs = ["arg_parse.cpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf",
        "//gxf/std",
        "@yaml-cpp",
        "arg",
        "proxy_component",
    ],
)

nv_gxf_cc_library(
    name = "extension_manager",
    hdrs = ["extension_manager.hpp"],
    srcs = ["extension_manager.cpp"],
    visibility = ["//visibility:public"],
    linkopts = ["-ldl"],
    deps = [
        "//gxf/core:gxf",
        "//gxf/std",
        "@yaml-cpp",
    ],
)

nv_gxf_cc_library(
    name = "proxy_component",
    hdrs = ["proxy_component.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf",
        "//gxf/std",
        "//gxf/sample",
        "@yaml-cpp",
        "arg",
    ],
)

nv_gxf_cc_library(
    name = "app",
    srcs = [
            "application.cpp",
            "driver.cpp",
            "worker.cpp",
        ],
    hdrs = [
            "application.hpp",
            "driver.hpp",
            "worker.hpp",
        ],
    visibility = ["//visibility:public"],
    deps = [
        "arg_parse",
        "config_parser",
        "entity_group",
        "extension_manager",
        "graph_entity",
        "graph_utils",
        "proxy_component",
        "segment",
        "//common",
        "//gxf/core:gxf",
        "//gxf/ipc/http",
        "//gxf/std",
        "//gxf/ucx",
        "@breakpad//:breakpad",
    ],
)

nv_gxf_cc_library(
    name = "segment",
    srcs = ["segment.cpp"],
    hdrs = ["segment.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "arg_parse",
        "entity_group",
        "graph_entity",
        "graph_utils",
        "proxy_component",
        "//gxf/core:gxf",
        "//gxf/std",
        "//gxf/ucx",
    ],
)

nv_gxf_cc_library(
    name = "graph_entity",
    srcs = ["graph_entity.cpp"],
    hdrs = [
            "graph_entity.hpp",
            "arg.hpp",
           ],
    visibility = ["//visibility:public"],
    deps = [
        "arg_parse",
        "//gxf/core:gxf",
        "//gxf/std",
        "//gxf/sample",
    ],
)

nv_gxf_cc_library(
    name = "entity_group",
    srcs = ["entity_group.cpp"],
    hdrs = ["entity_group.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "graph_entity",
    ],
)

nv_gxf_cc_library(
    name = "graph_utils",
    hdrs = ["graph_utils.hpp"],
    visibility = ["//visibility:public"],
)