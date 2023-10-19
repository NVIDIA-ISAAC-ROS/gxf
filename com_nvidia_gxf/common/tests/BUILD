"""
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

cc_test(
    name = "test_endian",
    srcs = ["test_endian.cpp"],
    deps = [
        "//common:endian",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_expected",
    srcs = ["test_expected.cpp"],
    deps = [
        "//common:expected",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_memory_utils",
    size = "small",
    srcs = ["test_memory_utils.cpp"],
    deps = [
        "//common:memory_utils",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_type_utils",
    size = "small",
    srcs = ["test_type_utils.cpp"],
    deps = [
        "//common:type_utils",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_unique_index_map",
    size = "small",
    srcs = ["test_unique_index_map.cpp"],
    deps = [
        "//common:unique_index_map",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_type_name",
    srcs = ["test_type_name.cpp"],
    deps = [
        "//common:type_name",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_fixed_vector",
    srcs = ["test_fixed_vector.cpp"],
    deps = [
        "//common:fixed_vector",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_fixed_map",
    srcs = ["test_fixed_map.cpp"],
    deps = [
        "//common:fixed_map",
        "//common:fixed_string",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_fixed_string",
    srcs = ["test_fixed_string.cpp"],
    deps = [
        "//common:fixed_string",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_strong_type",
    srcs = ["test_strong_type.cpp"],
    deps = [
        "//common:strong_type",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_iterator",
    srcs = ["test_iterator.cpp"],
    deps = [
        "//common:iterator",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_span",
    srcs = ["test_span.cpp"],
    deps = [
        "//common:span",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_logger",
    srcs = ["test_logger.cpp"],
    deps = [
        "//common:logger",
        "@gtest//:main",
    ],
)
