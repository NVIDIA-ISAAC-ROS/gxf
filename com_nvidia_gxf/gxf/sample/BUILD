"""
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

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
load("//gxf:gxf.bzl", "nv_gxf_components_pyclass_gen")

exports_files([
    "__init__.py",
])

# nv_gxf_components_pyclass_gen(
#     name = "pyModuleGen",
#     visibility = ["//visibility:public"],
#     extensions = ["//gxf/sample"],
# )

nv_gxf_cc_extension(
    name = "sample",
    srcs = ["sample.cpp"],
    deps = [
        "hello_world",
        "ping_rx",
        "ping_rx_async",
        "ping_tx",
        "ping_tx_async",
        "ping_batch_rx",
        "multi_ping_rx",
        "//gxf/core:gxf",
        "//gxf/std:extension_factory_helper",
    ],
    ext_deps = [
        "//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "ping_tx",
    srcs = ["ping_tx.cpp"],
    hdrs = ["ping_tx.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/std:clock",
        "//gxf/std:codelet",
        "//gxf/std:resources",
    ],
)

nv_gxf_cc_library(
    name = "ping_tx_async",
    srcs = ["ping_tx_async.cpp"],
    hdrs = ["ping_tx_async.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/std:codelet",
    ],
)

nv_gxf_cc_library(
    name = "ping_rx",
    srcs = ["ping_rx.cpp"],
    hdrs = ["ping_rx.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/std:codelet",
        "//gxf/std:resources",
    ],
)

nv_gxf_cc_library(
    name = "ping_rx_async",
    srcs = ["ping_rx_async.cpp"],
    hdrs = ["ping_rx_async.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/std:codelet",
    ],
)

nv_gxf_cc_library(
    name = "ping_batch_rx",
    srcs = ["ping_batch_rx.cpp"],
    hdrs = ["ping_batch_rx.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/std:codelet",
        "//gxf/std:resources",
    ],
)

nv_gxf_cc_library(
    name = "multi_ping_rx",
    srcs = ["multi_ping_rx.cpp"],
    hdrs = ["multi_ping_rx.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:parameter_parser_std",
        "//gxf/std:codelet",
    ],
)

nv_gxf_cc_library(
    name = "hello_world",
    hdrs = ["hello_world.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/std:codelet",
    ],
)

py_library(
    name = "sample_extension",
    srcs = [
        "__init__.py",
        "Components.py"
    ],
    visibility = ["//visibility:public"],
    data = [
        "//gxf/sample:Components.py",
    ],
)

load("//registry/build:registry.bzl", "register_extension")

exports_files(["LICENSE"])

register_extension(
    name = "register_sample_ext",
    badges = [""],
    extension = "sample",
    headers = [
        "BUILD.public",
        "hello_world.hpp",
        "ping_rx.hpp",
        "ping_rx_async.hpp",
        "ping_tx.hpp",
        "ping_tx_async.hpp",
        "ping_batch_rx.hpp",
    ],
    python_sources = [
        "//gxf/sample:__init__.py",
        "//gxf/sample:Components.py",
    ],
    labels = ["gxf"],
    license_file = ":LICENSE",
    local_dependencies = ["//gxf/std:register_std_ext"],
    priority = "1",
    url = "www.nvidia.com",
    uuid = "a6ad78b6-1682-11ec-9621-0242ac130002",
    version = "1.6.0",
    visibility = ["//visibility:public"],
)
