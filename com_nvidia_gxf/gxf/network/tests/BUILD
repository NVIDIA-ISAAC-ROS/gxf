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
    "nv_gxf_app",
    "nv_gxf_cc_extension",
    "nv_gxf_cc_library",
)

cc_test(
    name = "test_clock_sync",
    srcs = ["test_clock_sync.cpp"],
    data = [
        "//gxf/network:libgxf_network.so",
        "//gxf/network/tests:libgxf_test_clock_sync_helpers_factory.so",
        "//gxf/network/tests:test_clock_sync_primary.yaml",
        "//gxf/network/tests:test_clock_sync_secondary.yaml",
        "//gxf/serialization:libgxf_serialization.so",
        "//gxf/std:libgxf_std.so",
        "//gxf/test/extensions:libgxf_test.so",
    ],
    # Changing it to manual
    tags = ["manual"],  # FIXME: flakey on CI/CD
    deps = [
        "test_clock_sync_helpers",
        "//gxf/core:gxf",
        "//gxf/network:clock_sync_primary",
        "//gxf/network:clock_sync_secondary",
        "//gxf/std",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_tcp",
    srcs = ["test_tcp.cpp"],
    data = [
        "//gxf/network:libgxf_network.so",
        "//gxf/network/tests:test_tcp_client_bidirectional.yaml",
        "//gxf/network/tests:test_tcp_client_reconnection_server.yaml",
        "//gxf/network/tests:test_tcp_client_sink.yaml",
        "//gxf/network/tests:test_tcp_client_sink_async.yaml",
        "//gxf/network/tests:test_tcp_client_source.yaml",
        "//gxf/network/tests:test_tcp_client_stress.yaml",
        "//gxf/network/tests:test_tcp_server_bidirectional.yaml",
        "//gxf/network/tests:test_tcp_server_reconnection_client.yaml",
        "//gxf/network/tests:test_tcp_server_sink.yaml",
        "//gxf/network/tests:test_tcp_server_source.yaml",
        "//gxf/network/tests:test_tcp_server_source_async.yaml",
        "//gxf/network/tests:test_tcp_server_stress.yaml",
        "//gxf/network/tests:test_tcp_server_receiver_push_server.yaml",
        "//gxf/network/tests:test_tcp_server_receiver_push_client.yaml",
        "//gxf/sample:libgxf_sample.so",
        "//gxf/serialization:libgxf_serialization.so",
        "//gxf/std:libgxf_std.so",
        "//gxf/test/extensions:libgxf_test.so",
    ],
    # Changing it to manual
    tags = ["manual"],  # FIXME: flakey on CI/CD
    deps = [
        "//gxf/core:gxf",
        "//gxf/std:tensor",
        "//gxf/std:double_buffer_receiver",
        "//gxf/test/components:components",
        "//gxf/network:tcp_client_socket",
        "//gxf/network:tcp_server_socket",
        "@gtest//:main",
    ],
)

nv_gxf_app(
    name = "test_tcp_client_bidirectional_stress",
    srcs = ["test_tcp_client_bidirectional_stress.yaml"],
    extensions = [
        "//gxf/std",
        "//gxf/serialization",
        "//gxf/test/extensions:test",
        "//gxf/network",
    ],
)

nv_gxf_app(
    name = "test_tcp_server_bidirectional_stress",
    srcs = ["test_tcp_server_bidirectional_stress.yaml"],
    extensions = [
        "//gxf/std",
        "//gxf/serialization",
        "//gxf/test/extensions:test",
        "//gxf/network",
    ],
)

nv_gxf_cc_extension(
    name = "test_clock_sync_helpers_factory",
    srcs = ["test_clock_sync_helpers_factory.cpp"],
    ext_deps = ["//gxf/std"],
    deps = [
        "test_clock_sync_helpers",
    ],
)

nv_gxf_cc_library(
    name = "test_clock_sync_helpers",
    srcs = ["test_clock_sync_helpers.cpp"],
    hdrs = ["test_clock_sync_helpers.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/std",
    ],
)
