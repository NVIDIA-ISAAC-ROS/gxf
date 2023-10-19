"""
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

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
    "nv_gxf_cc_test_group",
    "nv_gxf_app"
)


load("//registry/build:registry.bzl", "register_extension")

cc_test(
    name = "test_cuda_unit",
    srcs = ["test_cuda_unit.cpp"],
    data = [
        "test_cuda_unit.yaml",
        "//gxf/gxe:manifest",
    ],
    deps = [
        "//gxf/core:gxf",
        "//gxf/cuda",
        "//gxf/std",
        "//gxf/cuda:cuda_event",
        "//gxf/cuda:cuda_stream_id",
        "//gxf/cuda:cuda_stream_pool",
        "//common",
        "//third_party:cudart",
        "@yaml-cpp",
        "@gtest//:main",
    ],
)

nv_gxf_cc_test_group(
    name = "tests",
    apps = [
        "gxf/cuda/tests/test_cuda_stream_dotproduct.yaml",
    ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
    ],
    manifest_path_hack = "gxf/cuda/tests/",
    deps = ["//gxf/test:test_main"],
)

nv_gxf_cc_extension(
    name = "test_cuda",
    srcs = ["test_cuda_ext.cpp"],
    deps = [
        "test_cuda_helper",
        "//gxf/core:gxf",
        "//gxf/std:tensor",
    ],
    ext_deps = [
        "//gxf/std",
        "//gxf/cuda",
    ],
)

register_extension(
    name = "register_test_cuda_ext",
    badges = [""],
    extension = "test_cuda",
    uuid = "469d9214-46a5-431a-b7b5-9e598d9fc1db",
    version = "1.3.0",
    headers = [
        "test_cuda_helper.hpp"
    ],
    labels = ["gxf"],
    license_file = ":LICENSE",
    url = "www.nvidia.com",
    compute_dependencies = ["cuda"],
    local_dependencies = ["//gxf/std:register_std_ext",
                          "//gxf/cuda:register_cuda_ext"],
    priority = "1",
    visibility = ["//visibility:public"],
)

nv_gxf_cc_library(
    name = "test_cuda_helper",
    hdrs = ["test_cuda_helper.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/cuda:cuda_event",
        "//gxf/cuda:cuda_stream",
        "//gxf/cuda:cuda_stream_id",
        "//gxf/cuda:cuda_stream_pool",
        "//gxf/cuda:cuda_common",
        "//gxf/std",
        "//third_party:cublas",
        "//third_party:cudart",
        "//common",
        "@yaml-cpp"
    ],
)

test_stress_cuda_stream_dotproduct_yaml_files = [
    "test_stress_cuda_stream_dotproduct_gen.yaml",
]

genrule(
    name = "test_stress_cuda_stream_dotproduct_gen",
    outs = test_stress_cuda_stream_dotproduct_yaml_files,
    cmd = "$(location :test_stress_cuda_stream_dotproduct) $(RULEDIR)",
    message = "----- Generating the test_stress_cuda_stream_dotproduct_gen.yaml file",
    tools = [":test_stress_cuda_stream_dotproduct"],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "test_stress_cuda_stream_dotproduct",
    srcs = ["test_stress_cuda_stream_dotproduct.py"],
)

nv_gxf_app(
    name = "test_stress_cuda_stream_dotproduct_app",
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_stress_cuda_stream_dotproduct_gen.yaml",
)