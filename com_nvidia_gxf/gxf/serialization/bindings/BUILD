"""
Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

exports_files([
    "__init__.py",
    "cask.py"
])

load("//gxf:pybind.bzl", "pybind_library")


py_library(
    name = "serialization",
    srcs = [
        "//gxf/serialization/bindings:__init__.py",
        "//gxf/serialization/bindings:cask.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:core_pybind",
        "//gxf/std:tensor_pybind",
        "//gxf/std:vault_pybind"
    ],
)