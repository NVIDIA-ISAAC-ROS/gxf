"""
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//gxf/python:pygxf.bzl", "nv_pygxf_test")

nv_pygxf_test(
    name = "test_tensor_apps",
    srcs = ["test_tensor_apps.py"],
    app = "gxf/python/tests/dlpack/test_tensor_apps.py",
    data = [
        "MatMul.py",
        "StepCount.py",
        "TensorPairGenerator.py",
        "VerifyEqual.py",
        "generate_dlpack_apps.py",
    ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/python_codelet",
        "//gxf/test/extensions:test",
    ],
    manifest_path_hack = "gxf/python/tests/dlpack/",
    tags = ["pytest"],
    deps = [
        "//gxf/cuda:cuda_extension",
        "//gxf/python_codelet:python_codelet_extension",
        "//gxf/std:std_extension",
    ],
)

nv_pygxf_test(
    name = "test_python_api_dlpack",
    srcs = ["test_python_api_dlpack.py"],
    app = "gxf/python/tests/dlpack/test_python_api_dlpack.py",
    data = [
        "MatMul.py",
        "StepCount.py",
        "TensorPairGenerator.py",
        "VerifyEqual.py",
    ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/python_codelet",
    ],
    manifest_path_hack = "gxf/python/tests/dlpack/",
    tags = ["pytest"],
    deps = [
        "//gxf/cuda:cuda_extension",
        "//gxf/python",
        "//gxf/python_codelet:python_codelet_extension",
        "//gxf/std:std_extension",
    ],
)
