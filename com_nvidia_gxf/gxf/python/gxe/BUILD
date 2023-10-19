"""
Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//gxf/python:pygxf.bzl", "nv_pygxf_app")

exports_files([
    "gxe.py",
])

nv_pygxf_app(
    name = "gxe",
    srcs = ["gxe.py"],
    app = "gxf/python/gxe/test_core.yaml",
    extensions = [
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/python_codelet",
        "//gxf/sample",
    ],
    manifest_path_hack = "gxf/python/gxe/",
    #FIXME: move these dependencies to manifest file
    deps = [
        "//gxf/python",
        "//gxf/python_codelet:python_codelet_extension",
        "//gxf/cuda:cuda_extension",
        "//gxf/std:std_extension"
    ],
    data = [
        "//gxf/python/tests:python_test_codelet",
    ],
)
