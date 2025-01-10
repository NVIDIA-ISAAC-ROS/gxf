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
    "nv_gxf_cc_extension",
    "nv_gxf_cc_library",
)

load("//gxf:pybind.bzl", "pybind_library")

exports_files([
    "__init__.py",
])

py_library(
    name = "rmm_extension",
    srcs = [
        "__init__.py",
        "Components.py",
    ],
    visibility = ["//visibility:public"]
)

nv_gxf_cc_library(
    name = "rmm_lib",
    visibility = ["//visibility:public"],
    deps = [
        "rmm_allocator",
        "//gxf/core:gxf",
    ],
)

nv_gxf_cc_extension(
    name = "rmm",
    srcs = ["rmm.cpp"],
    deps = [
        "rmm_lib",
        "//gxf/std:extension_factory_helper",
    ],
    ext_deps = ["//gxf/std"],
)

nv_gxf_cc_library(
    name = "rmm_allocator",
    srcs = ["rmm_allocator.cpp"],
    hdrs = ["rmm_allocator.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf",
        "//gxf/cuda:cuda_allocator",
        "//gxf/cuda:cuda_common",
        "//third_party:cudart",
        "@rmm//:rmm",
    ],
)

load("//registry/build:registry.bzl", "register_extension")

exports_files(["LICENSE"])

register_extension(
    name = "register_rmm_ext",
    badges = [""],
    extension = "rmm",
    uuid = "45fa2002-1f83-11ef-a380-ef5f83754f84",
    labels = ["gxf"],
    license_file = ":LICENSE",
    python_sources = [
        "//gxf/rmm:__init__.py",
        "//gxf/rmm:Components.py",
    ],
    local_dependencies = ["//gxf/std:register_std_ext",
                          "//gxf/cuda:register_cuda_ext"],
    priority = "1",
    url = "www.nvidia.com",
    version = "0.0.1",
    visibility = ["//visibility:public"],
)
