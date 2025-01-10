"""
Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

# https://docs.bazel.build/versions/master/be/c-cpp.html#cc_library
cc_library(
    name = "test_load_extension",
    # srcs = ["source files"],
    hdrs = ["test_load_extension.hpp"],
    visibility = ["//visibility:public"],
)
cc_test(
    name = "test_gxf_core",
    srcs = [
        "test_gxf_core.cpp",
        "test_load_extension.hpp"
    ],
    data = [
         "test_ping_expiring_message_available.yaml",
        "test_push_to_transmitter.yaml",
        "//gxf/gxe:manifest",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:test_ping_graph.yaml",
        "//gxf/test/apps:test_ping.yaml",
        "//gxf/test/apps:test_ping_multi_thread_start_order.yaml",
        "//gxf/test/apps:test_metric.yaml",
        "//gxf/test/apps:test_ping_unnamed_entities.yaml",
        "//gxf/test/apps:test_ping_unnamed_entities_without_scheduler.yaml"
    ],
    deps = [
        "//gxf/core:gxf",
        "//gxf/std",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_gxf_clock",
    srcs = ["test_gxf_clock.cpp"],
    data = [
        "test_realtime_clock.yaml",
        "//gxf/gxe:manifest",
    ],
    deps = [
        "//gxf/core:gxf",
        "//gxf/std",
        "@gtest//:main",
    ],
)


cc_test(
    name = "test_gxf_scheduling_terms",
    srcs = ["test_gxf_scheduling_terms.cpp"],
    data = [
        "test_ping_expiring_message_available.yaml",
        "//gxf/gxe:manifest",
    ],
    deps = [
        "//gxf/core:gxf",
        "//gxf/std",
        "@gtest//:main",
    ],
    tags = ["performance"],  # FIXME: flakey on CI/CD
)

cc_test(
    name = "test_gxf_std_vault",
    srcs = [
        "test_gxf_std_vault.cpp",
    ],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:test_vault.yaml"
    ],
    deps = [
        "//gxf/core:gxf",
        "//gxf/std",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_gxf_entity_group",
    srcs = [
        "test_gxf_entity_group.cpp",
    ],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/test/apps:test_entity_group_default.yaml",
        "//gxf/test/apps:test_entity_group_users.yaml",
        "//gxf/test/apps:test_entity_group_default_and_users.yaml",
        "//gxf/test/apps:entity_group_subgraph.yaml",
        "//gxf/test/apps:test_entity_group_root_and_subgraph.yaml",
    ],
    deps = [
        "//gxf/core:gxf",
        "//gxf/sample",
        "//gxf/std",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_entity_thread_safe",
    srcs = [
        "test_entity_thread_safe.cpp",
    ],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/std",
        "//gxf/test/extensions:test",
    ],
    deps = [
        "//gxf/core:gxf",
        "//gxf/std",
        "@gtest//:main",
    ],
)

# Adding test extension registration hook here since //gxf/test/extensions/BUILD file
# is also used in release package while the test extension itself is not included in the
# release package

load("//registry/build:registry.bzl", "register_extension")
exports_files(["LICENSE"])

register_extension(
    name = "register_test_ext",
    extension = "//gxf/test/extensions:test",
    uuid = "1b99ffeb-c250-4ced-8117-62ac05969a50",
    version = "2.6.0",
    license_file = "//gxf/test/extensions:LICENSE",
    url = "www.nvidia.com",
    labels = ["gxf"],
    badges = [""],
    priority = "1",
    local_dependencies = ["//gxf/std:register_std_ext"],
    compute_dependencies = ["cuda"],
    visibility = ["//visibility:public"],
    python_alias = "test",
)
