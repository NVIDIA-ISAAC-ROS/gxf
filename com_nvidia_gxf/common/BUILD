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
    "nv_gxf_cc_library",
)

load("@com_nvidia_gxf//packaging:packaging.bzl", "nv_gxf_pkg_library")

nv_gxf_cc_library(
    name = "assert",
    hdrs = ["assert.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "backtrace",
        "logger",
    ],
)

nv_gxf_cc_library(
    name = "backtrace",
    srcs = ["backtrace.cpp"],
    hdrs = ["backtrace.hpp"],
    visibility = ["//visibility:public"],
)

nv_gxf_cc_library(
    name = "byte",
    hdrs = ["byte.hpp"],
    visibility = ["//visibility:public"],
)

nv_gxf_cc_library(
    name = "endian",
    hdrs = ["endian.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "type_utils",
    ],
)

nv_gxf_cc_library(
    name = "expected",
    hdrs = ["expected.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "assert",
        "memory_utils",
        "strong_type",
        "type_utils",
    ],
)

nv_gxf_cc_library(
    name = "fixed_string",
    hdrs = ["fixed_string.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "expected",
        "iterator",
        "type_utils",
    ],
)

nv_gxf_cc_library(
    name = "fixed_vector",
    hdrs = ["fixed_vector.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "byte",
        "expected",
        "iterator",
        "memory_utils",
    ],
)

nv_gxf_cc_library(
    name = "fixed_map",
    hdrs = ["fixed_map.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "expected",
        "memory_utils",
    ],
)

nv_gxf_cc_library(
    name = "iterator",
    hdrs = ["iterator.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "expected",
        "type_utils",
    ],
)

nv_gxf_cc_library(
    name = "logger",
    srcs = ["logger.cpp"],
    hdrs = ["logger.hpp"],
    deps = ["singleton"],
    visibility = ["//visibility:public"],
)

nv_gxf_cc_library(
    name = "memory_utils",
    hdrs = ["memory_utils.hpp"],
    visibility = ["//visibility:public"],
    deps = ["byte"],
)

nv_gxf_cc_library(
    name = "nvtx_helper",
    hdrs = ["nvtx_helper.hpp"],
    visibility = ["//visibility:public"],
    deps = ["//third_party:nvtx"],
)

nv_gxf_cc_library(
    name = "singleton",
    srcs = ["singleton.hpp"],
    visibility = ["//visibility:public"],
)

nv_gxf_cc_library(
    name = "span",
    hdrs = ["span.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "byte",
        "expected",
        "iterator",
        "type_utils",
    ],
)

nv_gxf_cc_library(
    name = "strong_type",
    hdrs = ["strong_type.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "type_utils",
    ],
)

nv_gxf_cc_library(
    name = "type_name",
    srcs = [
        "type_name.cpp",
        "type_name_gnuc.hpp",
    ],
    hdrs = ["type_name.hpp"],
    visibility = ["//visibility:public"],
)

nv_gxf_cc_library(
    name = "type_utils",
    hdrs = ["type_utils.hpp"],
    visibility = ["//visibility:public"],
)

nv_gxf_cc_library(
    name = "unique_index_map",
    hdrs = ["unique_index_map.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "expected",
    ],
)

nv_gxf_cc_library(
    name = "common",
    visibility = ["//visibility:public"],
    deps = [
        "assert",
        "backtrace",
        "byte",
        "endian",
        "expected",
        "fixed_map",
        "fixed_string",
        "fixed_vector",
        "iterator",
        "logger",
        "memory_utils",
        "nvtx_helper",
        "singleton",
        "span",
        "strong_type",
        "type_name",
        "unique_index_map",
    ],
)

nv_gxf_pkg_library(
    name = "common",
    build = "BUILD.release",
    headers = [
        "assert.hpp",
        "backtrace.hpp",
        "byte.hpp",
        "endian.hpp",
        "expected.hpp",
        "fixed_map",
        "fixed_string.hpp",
        "fixed_vector.hpp",
        "iterator.hpp",
        "logger.hpp",
        "memory_utils.hpp",
        "nvtx_helper.hpp",
        "singleton.hpp",
        "span.hpp",
        "strong_type.hpp",
        "type_name.hpp",
        "type_name_gnuc.hpp",
        "type_utils.hpp",
    ],
)
