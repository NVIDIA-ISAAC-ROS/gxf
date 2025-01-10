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
    "nv_gxf_test_app",
    "nv_gxf_cc_test_group",
    "nv_gxf_app",
    "nv_gxf_cc_test",
)

load("//gxf/python:pygxf.bzl", "nv_pygxf_test")

nv_gxf_test_app(
    name = "test_rmm_allocator",
    srcs = ["test_rmm_allocator.yaml"],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/rmm",
        "//gxf/test/extensions:test",
    ]
)

nv_gxf_cc_test(
    name = "rmm_memory_pool",
    size = "small",
    srcs = ["test_memory_pool.cpp"],
    data = [
        "//gxf/gxe:manifest",
    ],
    deps = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/rmm:rmm",
        "//gxf/test/extensions:test",
        "@gtest//:main",
    ],
)

nv_gxf_test_app(
    name = "test_rmm_allocator_stress_test",
    srcs = ["test_rmm_allocator_stress_test.yaml"],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/rmm",
        "//gxf/test/extensions:test",
        "//gxf/sample:sample"
    ],
)

nv_gxf_cc_test_group(
    name = "tests",
    apps = [
        "gxf/rmm/tests/test_rmm_allocator_stress_test.yaml",
        "gxf/rmm/tests/test_cuda_stream_dotproduct_rmm_allocator.yaml",
        "gxf/rmm/tests/test_rmm_allocator_cuda_event_st_dotproduct.yaml"
    ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/rmm",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
        "//gxf/sample:sample",
    ],
    manifest_path_hack = "gxf/rmm/tests/",
    deps = ["//gxf/test:test_main"],
)

test_stress_cuda_stream_dotproduct_rmm_allocator_yaml_files = [
    "test_stress_cuda_stream_dotproduct_rmm_allocator_gen.yaml",
]

genrule(
    name = "test_stress_cuda_stream_dotproduct_rmm_allocator_gen",
    outs = test_stress_cuda_stream_dotproduct_rmm_allocator_yaml_files,
    cmd = "$(location :test_stress_cuda_stream_dotproduct_rmm_allocator) $(RULEDIR)",
    message = "----- Generating the test_stress_cuda_stream_dotproduct_rmm_allocator_gen.yaml file",
    tools = [":test_stress_cuda_stream_dotproduct_rmm_allocator"],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "test_stress_cuda_stream_dotproduct_rmm_allocator",
    srcs = ["test_stress_cuda_stream_dotproduct_rmm_allocator.py"],
)

nv_gxf_app(
    name = "test_stress_cuda_stream_dotproduct_rmm_allocator_app",
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/rmm",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_stress_cuda_stream_dotproduct_rmm_allocator_gen.yaml",
)

nv_pygxf_test(
    name = "test_video_bindings_with_rmm_allocator",
    srcs = ["test_video_bindings_with_rmm_allocator.py"],
    data = [],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/sample",
        "//gxf/python_codelet",
        "//gxf/multimedia",
        "//gxf/rmm"
    ],
    manifest_path_hack = "gxf/multimedia/tests/",
    tags = [
        "manual",
        "pytest",
    ],
    deps = [
        "//gxf/cuda:cuda_extension",
        "//gxf/python",
        "//gxf/python_codelet:python_codelet_extension",
        "//gxf/sample:sample_extension",
        "//gxf/std:std_extension",
        "//gxf/multimedia:multimedia_extension",
        "//gxf/rmm:rmm_extension"
    ],
)

