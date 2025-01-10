"""
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

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
)

cc_test(
    name = "test_gxf_stream_unit",
    size = "small",
    srcs = ["test_gxf_stream_unit.cpp"],
    data = [
        "//gxf/gxe:manifest",
    ],
    deps = [
        "//gxf/stream",
        "@gtest//:main",
    ],
)

nv_gxf_cc_test_group(
    name = "tests",
    apps = [
        "gxf/stream/tests/test_gxf_stream_cuda_dotproduct.yaml",
    ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/stream",
        "//gxf/stream/tests:test_stream_sync_cuda",
        "//gxf/test/extensions:test",
    ],
    manifest_path_hack = "gxf/stream/tests/",
    deps = [
        "//gxf/test:test_main",
        "//third_party:nvscisync",
    ],
)

nv_gxf_cc_extension(
    name = "test_stream_sync_cuda",
    srcs = ["test_stream_sync_cuda_ext.cpp"],
    deps = [
        "test_stream_sync_cuda_helper",
        "//gxf/std:tensor",
        "//gxf/std:extension_factory_helper",
        "//gxf/stream:stream_nvsci",
        "//gxf/stream",
    ],
    ext_deps = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/stream",
    ],
)

nv_gxf_cc_library(
    name = "test_stream_sync_cuda_helper",
    hdrs = ["test_gxf_stream_sync_cuda_helper.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/cuda",
        "//third_party:cublas",
    ],
)
