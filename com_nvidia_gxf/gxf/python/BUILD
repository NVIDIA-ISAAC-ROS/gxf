"""
Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

exports_files([
    "gxf/codelet.py",
    "gxf/core.py",
    "gxf/__init__.py",
    "gxf/serialization/__init__.py",
    "gxf/serialization/cask.py",
    "gxf/std/python/__init__.py",
    "gxf/python/tests/test_std_tensor.py",
    "gxf/python/tests/test_boolean_scheduling_term.py",
    "gxf/python/tests/test_count_scheduling_term.py",
    "gxf/python/tests/test_std_vault.py"
])


py_library(
    name = "python",
    srcs = [
        "//gxf/python:gxf/__init__.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:core_pybind",
    ],
)