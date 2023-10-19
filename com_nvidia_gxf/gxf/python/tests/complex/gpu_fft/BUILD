"""
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//gxf/python:pygxf.bzl", "nv_pygxf_test")

exports_files([
    "CreateTensor.py",
    "RunFFT.py",
    "VerifyTensor.py",
])

nv_pygxf_test(
    name = "test_gpu_fft",
    srcs = ["test_gpu_fft.py"],
    app = "gxf/python/tests/complex/gpu_fft/test_gpu_fft.yaml",
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/python_codelet",
        "//gxf/test/extensions:test",
    ],
    manifest_path_hack = "gxf/python/tests/complex/gpu_fft",
    tags = ["pytest", "manual"],
    deps = [
        "//gxf/cuda:cuda_extension",
        "//gxf/python",
        "//gxf/python_codelet:python_codelet_extension",
        "//gxf/std:std_extension"
    ],
    data = [
        "CreateTensor.py",
        "RunFFT.py",
        "VerifyTensor.py",
       "//gxf/python/tests:python_test_codelet",
    ],
)