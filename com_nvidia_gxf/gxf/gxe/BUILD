"""
Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load(
    "//gxf:gxf.bzl",
    "nv_gxf_cc_binary",
)

nv_gxf_cc_binary(
    name = "gxe",
    srcs = ["gxe.cpp"],
    data = [
        "manifest",
        "//gxf/test/apps:yaml_files",
    ],
    visibility = ["//visibility:public"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../core"],
    deps = [
        "//common",
        "//external:gflags",
        "//gxf/core:libgxf_core",
        "@breakpad//:breakpad"
    ],
)

filegroup(
    name = "manifest",
    srcs = ["manifest.yaml"],
    data = [
        "//gxf/behavior_tree:libgxf_behavior_tree.so",
        "//gxf/cuda:libgxf_cuda.so",
        "//gxf/cuda/tests:libgxf_test_cuda.so",
        "//gxf/multimedia:libgxf_multimedia.so",
        "//gxf/network:libgxf_network.so",
        "//gxf/npp:libgxf_npp.so",
        "//gxf/rmm:libgxf_rmm.so",
        "//gxf/sample:libgxf_sample.so",
        "//gxf/serialization:libgxf_serialization.so",
        "//gxf/std:libgxf_std.so",
        "//gxf/stream:libgxf_stream.so",
        "//gxf/stream/tests:libgxf_test_stream_sync_cuda.so",
        "//gxf/test/extensions:libgxf_test.so",
        "//gxf/ucx:libgxf_ucx.so",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "manifest_extended",
    srcs = ["manifest_extended.yaml"],
    data = [
        "//gxf/behavior_tree:libgxf_behavior_tree.so",
        "//gxf/cuda:libgxf_cuda.so",
        "//gxf/cuda/tests:libgxf_test_cuda.so",
        "//gxf/multimedia:libgxf_multimedia.so",
        "//gxf/network:libgxf_network.so",
        "//gxf/npp:libgxf_npp.so",
        "//gxf/sample:libgxf_sample.so",
        "//gxf/serialization:libgxf_serialization.so",
        "//gxf/std:libgxf_std.so",
        "//gxf/stream:libgxf_stream.so",
        "//gxf/stream/tests:libgxf_test_stream_sync_cuda.so",
        "//gxf/test/extensions:libgxf_test.so",
        "//gxf/ucx:libgxf_ucx.so",
        "//gxf/ipc/grpc:libgxf_grpc.so",
        "//gxf/ipc/http:libgxf_http.so"
    ],
    visibility = ["//visibility:public"],
)
