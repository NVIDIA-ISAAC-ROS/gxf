"""
Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

cc_test(
    name = "test_arg",
    srcs = ["test_arg.cpp"],
    deps = [
        "//gxf/core:gxf",
        "//gxf/app:arg",
        "//gxf/app:arg_parse",
        "@gtest//:main",
    ],
    data = [
        "//gxf/gxe:manifest",
    ]
)

cc_test(
    name = "test_graph_entity",
    srcs = ["test_graph_entity.cpp"],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/sample",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:yaml_files",
    ],
    deps = [
        "//gxf/app:graph_entity",
        "//gxf/core:gxf",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_entity_group",
    srcs = ["test_entity_group.cpp"],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/sample",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:yaml_files",
    ],
    deps = [
        "//gxf/app:entity_group",
        "//gxf/app:graph_entity",
        "//gxf/core:gxf",
        "@gtest//:main",
    ],
)

cc_binary(
    name = "test_ping_segment_distributed",
    srcs = [
        "//gxf/app/sample:distributed_headers",
        "multi_process_helper.hpp",
        "test_ping_segment_distributed.cpp",
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
    name = "test_ping_segment_distributed_graph",
    srcs = [
        "//gxf/app/sample:distributed_headers",
        "multi_process_helper.hpp",
        "test_ping_segment_distributed_graph.cpp"
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

cc_test(
    name = "ping_broadcast_tests",
    srcs = [
        "//gxf/app/sample:distributed_headers",
        "ping_broadcast_tests.cpp"
    ],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/gxe/tests:manifest_empty",
        "//gxf/gxe:manifest_extended",
        "//gxf/gxe/tests:manifest_invalid",
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
