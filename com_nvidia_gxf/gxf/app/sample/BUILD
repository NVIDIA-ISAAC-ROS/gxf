"""
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//gxf:gxf.bzl", "nv_gxf_cc_test")

distributed_headers = glob([
    "ping_segment_distributed.hpp",
    "ping_segment_distributed_graph.hpp",
])

filegroup(
    name = "distributed_headers",
    srcs = distributed_headers,
    visibility = ["//visibility:public"],
)

nv_gxf_cc_test(
    name = "ping_graph_runtime",
    srcs = ["ping_graph_runtime.cpp"],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/sample",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:yaml_files",
    ],
    deps = [
        "//common",
        "//gxf/app",
        "//gxf/app:graph_entity",
        "//gxf/core:gxf",
        "@gtest//:main",
    ],
)

nv_gxf_cc_test(
    name = "ping_graph_simple",
    srcs = ["ping_graph_simple.cpp"],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/sample",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:yaml_files",
    ],
    deps = [
        "//common",
        "//gxf/app",
        "//gxf/app:graph_entity",
        "//gxf/core:gxf",
        "@gtest//:main",
    ],
)


nv_gxf_cc_test(
    name = "ping_graph_multi",
    srcs = ["ping_graph_multi.cpp"],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/sample",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:yaml_files",
    ],
    deps = [
        "//common",
        "//gxf/app",
        "//gxf/app:graph_entity",
        "//gxf/test/extensions:test",
        "//gxf/core:gxf",
        "@gtest//:main",
    ],
)

nv_gxf_cc_test(
    name = "ping_graph_multi_gather",
    srcs = ["ping_graph_multi_gather.cpp"],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/sample",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:yaml_files",
    ],
    deps = [
        "//common",
        "//gxf/app",
        "//gxf/app:graph_entity",
        "//gxf/core:gxf",
        "@gtest//:main",
    ],
)

nv_gxf_cc_test(
    name = "ping_graph_multi_broadcast",
    srcs = ["ping_graph_multi_broadcast.cpp"],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/sample",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:yaml_files",
    ],
    deps = [
        "//common",
        "//gxf/app",
        "//gxf/app:graph_entity",
        "//gxf/core:gxf",
        "@gtest//:main",
    ],
)

nv_gxf_cc_test(
    name = "tensor_copier_host_to_device",
    srcs = ["tensor_copier_host_to_device.cpp"],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/sample",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:yaml_files",
        "tensor_copier_parameters.yaml",
    ],
    deps = [
        "//common",
        "//gxf/app",
        "//gxf/app:segment",
        "//gxf/app:graph_entity",
        "//gxf/core:gxf",
        "//gxf/test/extensions:test",
        "@gtest//:main",
    ],
)

nv_gxf_cc_test(
    name = "crash_backtrace_print",
    srcs = ["crash_backtrace_print.cpp"],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/sample",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:yaml_files",
    ],
    deps = [
        "//common",
        "//gxf/app",
        "//gxf/app:graph_entity",
        "//gxf/core:gxf",
        "@gtest//:main",
    ],
)

nv_gxf_cc_test(
    name = "entity_group_and_resource",
    srcs = ["entity_group_and_resource.cpp"],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/sample",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:yaml_files",
    ],
    deps = [
        "//common",
        "//gxf/app",
        "//gxf/app:entity_group",
        "//gxf/app:graph_entity",
        "//gxf/core:gxf",
        "@gtest//:main",
    ],
)

# Multi segment application unit tests
#

cc_binary(
    name = "ping_segment_distributed",
    srcs = [
        "ping_segment_distributed.hpp",
        "ping_segment_distributed.cpp",
    ],
    data = [
        "//gxf/gxe:manifest_extended",
        "//gxf/sample",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:yaml_files",
    ],
    deps = [
        "//gxf/app",
        "//gxf/app:segment",
        "//gxf/app:graph_entity",
        "//gxf/core:gxf",
    ],
    tags = ["exclusive", "requires-network"],
)

cc_binary(
    name = "ping_segment_distributed_graph",
    srcs = [
        "ping_segment_distributed_graph.hpp",
        "ping_segment_distributed_graph.cpp"
    ],
    data = [
        "//gxf/gxe:manifest_extended",
        "//gxf/sample",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:yaml_files",
    ],
    deps = [
        "//gxf/app",
        "//gxf/app:segment",
        "//gxf/app:graph_entity",
        "//gxf/core:gxf",
    ],
    tags = ["exclusive", "requires-network"],
)

nv_gxf_cc_test(
    name = "ping_multi_segment_example",
    srcs = ["ping_multi_segment_example.cpp"],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/sample",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:yaml_files",
    ],
    deps = [
        "//common",
        "//gxf/app",
        "//gxf/app:segment",
        "//gxf/app:graph_entity",
        "//gxf/core:gxf",
        "@gtest//:main",
    ],
    tags = ["exclusive"],
)

nv_gxf_cc_test(
    name = "ping_segment_example",
    srcs = ["ping_segment_example.cpp"],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/sample",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:yaml_files",
    ],
    deps = [
        "//common",
        "//gxf/app",
        "//gxf/app:segment",
        "//gxf/app:graph_entity",
        "//gxf/core:gxf",
        "@gtest//:main",
    ],
    tags = ["exclusive"],
)

nv_gxf_cc_test(
    name = "tensor_copier_device_to_system",
    srcs = ["tensor_copier_device_to_system.cpp"],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/sample",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:yaml_files",
    ],
    deps = [
        "//common",
        "//gxf/app",
        "//gxf/app:segment",
        "//gxf/app:graph_entity",
        "//gxf/core:gxf",
        "//gxf/test/extensions:test",
        "@gtest//:main",
    ],
)

nv_gxf_cc_test(
    name = "tensor_copier_host_to_system",
    srcs = ["tensor_copier_host_to_system.cpp"],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/sample",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:yaml_files",
    ],
    deps = [
        "//common",
        "//gxf/app",
        "//gxf/app:segment",
        "//gxf/app:graph_entity",
        "//gxf/core:gxf",
        "//gxf/test/extensions:test",
        "@gtest//:main",
    ],
)

nv_gxf_cc_test(
    name = "tensor_copier_system_to_device",
    srcs = ["tensor_copier_system_to_device.cpp"],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/sample",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:yaml_files",
    ],
    deps = [
        "//common",
        "//gxf/app",
        "//gxf/app:segment",
        "//gxf/app:graph_entity",
        "//gxf/core:gxf",
        "//gxf/test/extensions:test",
        "@gtest//:main",
    ],
)

nv_gxf_cc_test(
    name = "tensor_copier_system_to_host",
    srcs = ["tensor_copier_system_to_host.cpp"],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/sample",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:yaml_files",
    ],
    deps = [
        "//common",
        "//gxf/app",
        "//gxf/app:segment",
        "//gxf/app:graph_entity",
        "//gxf/core:gxf",
        "//gxf/test/extensions:test",
        "@gtest//:main",
    ],
)

nv_gxf_cc_test(
    name = "test_factorial",
    srcs = ["test_factorial.cpp"],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/sample",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:yaml_files",
    ],
    deps = [
        "//common",
        "//gxf/app",
        "//gxf/app:segment",
        "//gxf/app:graph_entity",
        "//gxf/core:gxf",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "@gtest//:main",
    ],
)

nv_gxf_cc_test(
    name = "tensor_copier_device_to_host",
    srcs = ["tensor_copier_device_to_host.cpp"],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/sample",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:yaml_files",
        "tensor_copier_parameters.yaml",
    ],
    deps = [
        "//gxf/app",
        "//gxf/app:segment",
        "//gxf/app:graph_entity",
        "//gxf/core:gxf",
        "//gxf/test/extensions:test",
        "@gtest//:main",
    ],
)

nv_gxf_cc_test(
    name = "test_hello_world",
    srcs = ["test_hello_world.cpp"],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/sample",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:yaml_files",
    ],
    deps = [
        "//gxf/app",
        "//gxf/app:graph_entity",
        "//gxf/test/extensions:test",
        "//gxf/core:gxf",
        "@gtest//:main",
    ],
)

nv_gxf_cc_test(
    name = "test_hello_world_multi",
    srcs = ["test_hello_world_multi.cpp"],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/sample",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:yaml_files",
    ],
    deps = [
        "//gxf/app",
        "//gxf/app:graph_entity",
        "//gxf/test/extensions:test",
        "//gxf/core:gxf",
        "@gtest//:main",
    ],
)
