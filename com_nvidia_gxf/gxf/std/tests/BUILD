"""
Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load(
    "//gxf:gxf.bzl",
    "nv_gxf_cc_library",
)

cc_test(
    name = "memory_pool",
    size = "small",
    srcs = ["test_memory_pool.cpp"],
    data = [
        "//gxf/gxe:manifest",
    ],
    deps = [
        "//gxf/std",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_unbounded_allocator",
    size = "small",
    srcs = ["test_unbounded_allocator.cpp"],
    data = [
        "//gxf/gxe:manifest",
    ],
    deps = [
        "//gxf/std",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_primitive_types",
    size = "small",
    srcs = ["test_primitive_types.cpp"],
    data = [
        "//gxf/std:libgxf_std.so",
    ],
    deps = [
        "//gxf/std",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_tensor",
    size = "small",
    srcs = ["test_tensor.cpp"],
    data = [
        "//gxf/std:libgxf_std.so",
    ],
    deps = [
        "//gxf/std",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_typed_message_view",
    size = "small",
    srcs = ["test_typed_message_view.cpp"],
    data = [
        "//gxf/std:libgxf_std.so",
    ],
    deps = [
        "test_typed_message_view_helper",
        "//gxf/std",
        "@gtest//:main",
    ],
)

nv_gxf_cc_library(
    name = "test_typed_message_view_helper",
    hdrs = ["test_typed_message_view_helper.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/std",
    ],
)

cc_test(
    name = "test_boolean_scheduling_term",
    size = "small",
    srcs = ["test_boolean_scheduling_term.cpp"],
    data = [
        "//gxf/std:libgxf_std.so",
    ],
    deps = [
        "//gxf/std",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_memory_available_scheduling_term",
    size = "small",
    srcs = ["test_memory_available_scheduling_term.cpp"],
    data = [
        "//gxf/std:libgxf_std.so",
    ],
    deps = [
        "//gxf/std",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_eos",
    size = "small",
    srcs = ["test_eos.cpp"],
    data = [
        "//gxf/std:libgxf_std.so",
    ],
    deps = [
        "//gxf/std",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_epoch_scheduler",
    size = "small",
    srcs = ["test_epoch_scheduler.cpp"],
    data = [
        "test_epoch_scheduler_app.yaml",
        "//gxf/sample:libgxf_sample.so",
        "//gxf/std:libgxf_std.so",
        "//gxf/test/extensions:libgxf_test.so",
    ],
    deps = [
        "//gxf/std",
        "@gtest//:main",
        "@yaml-cpp",
    ],
)

cc_test(
    name = "test_synthetic_clock",
    srcs = ["test_synthetic_clock.cpp"],
    data = [
        "//gxf/std:libgxf_std.so",
    ],
    deps = [
        "//gxf/std",
        "//gxf/std:synthetic_clock",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_multisource_timestamp",
    srcs = ["test_multisource_timestamp.cpp"],
    data = [
        "//gxf/std:libgxf_std.so",
    ],
    deps = [
        "//gxf/std",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_scheduler",
    srcs = ["test_scheduler.cpp"],
    data = [
        "//gxf/std:libgxf_std.so",
        "//gxf/test/extensions:libgxf_test.so",
        "//gxf/std/tests/apps:yaml_files"
    ],
    deps = [
        "//gxf/std",
        "@gtest//:main",
    ],
)