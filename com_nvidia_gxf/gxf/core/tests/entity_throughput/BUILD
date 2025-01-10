"""
Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

cc_test(
    name = "test_create_entity_rate",
    srcs = [
        "test_create_entity_rate.cpp",
    ],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/std",
        "//gxf/test/extensions:test",
    ],
    deps = [
        "//gxf/core:gxf",
        "//gxf/std",
        "//gxf/std/gems/utils:time",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_entity_create",
    srcs = ["test_create_entities.cpp",
    ],
    data = [
             "//gxf/gxe:manifest",
    ],
    deps = [
        "//gxf/core:gxf",
        "//gxf/std",
        "//gxf/std/gems/utils:time",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_entity_create_int",
    srcs = [
        "test_create_entities_int.cpp",
    ],
    data = [
             "//gxf/gxe:manifest",
    ],
    deps = [
        "//gxf/core:gxf",
        "//gxf/std",
        "//gxf/std/gems/utils:time",
        "//gxf/test/unit:test_load_extension",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_entity_create_int_ten",
    srcs = ["test_create_entities_int_ten.cpp",
    ],
    data = [
             "//gxf/gxe:manifest",
    ],
    deps = [
        "//gxf/core:gxf",
        "//gxf/std",
        "//gxf/std/gems/utils:time",
        "//gxf/test/unit:test_load_extension",
        "@gtest//:main",
    ],
)

