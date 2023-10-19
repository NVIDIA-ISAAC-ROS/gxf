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

load("//gxf:pybind.bzl", "pybind_library")

exports_files([
    "__init__.py",
    "codelet.py"
])

py_library(
    name = "python_codelet_extension",
    srcs = [
        "__init__.py",
        "codelet.py",
        "Components.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/python_codelet:pycodelet",
        "//gxf/core:core_pybind",
        "//gxf/std:tensor_pybind",
        "//gxf/std:vault_pybind"
    ],
)

nv_gxf_cc_library(
    name = "py_codelet",
    srcs = ["py_codelet.cpp"],
    hdrs = ["py_codelet.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/cuda:cuda_stream_id",
        "//gxf/cuda:cuda_stream_pool",
        "//gxf/std:allocator",
        "//gxf/std:clock",
        "//gxf/std:codelet",
        "//gxf/std:parameter_parser_std",
        "//gxf/std:scheduling_terms",
        "//gxf/std:receiver",
        "//gxf/std:transmitter",
        "@pybind11",
        "//third_party:python",
    ],
)

nv_gxf_cc_extension(
    name = "python_codelet",
    srcs = ["python_codelet.cpp"],
    visibility = ["//visibility:public"],
    deps = [
        "py_codelet",
    ],
    ext_deps = [
        "//gxf/std",
        "//gxf/cuda",
    ],
)

pybind_library(
    name = "pycodelet",
    cc_deps = [
        "//gxf/core:gxf",
        "//gxf/python_codelet",
        "//gxf/std",
    ],
    cc_srcs = [
        "bindings/pycodelet.cpp",
    ],
)

load("//registry/build:registry.bzl", "register_extension")

exports_files(["LICENSE"])

register_extension(
    name = "register_python_codelet_ext",
    extension = "python_codelet",
    headers = [
        "BUILD.public",
        "py_codelet.hpp"
        ],
    labels = ["gxf"],
    python_bindings = [
        "//gxf/python_codelet:pycodelet.so",
    ],
    python_sources = [
        "//gxf/python_codelet:__init__.py",
        "//gxf/python_codelet:Components.py",
        "codelet.py"],
    badges = [""],
    license_file = ":LICENSE",
    priority = "1",
    url = "www.nvidia.com",
    uuid = "787daddc-1c34-11ec-9621-0242ac130002",
    version = "0.3.0",
    local_dependencies = ["//gxf/std:register_std_ext",
                          "//gxf/cuda:register_cuda_ext"],
    compute_dependencies = ["cuda"],
    visibility = ["//visibility:public"],
)
