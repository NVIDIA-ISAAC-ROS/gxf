"""
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load(
    "//gxf:gxf.bzl",
    "nv_gxf_cc_test_group",
    "nv_gxf_cc_extension",
    "nv_gxf_cc_library",
    "nv_gxf_test_app",
)

cc_test(
    name = "test_entity_record_replay",
    srcs = ["test_entity_record_replay.cpp"],
    data = [
        "//gxf/gxe:manifest",
        "//gxf/serialization/tests:test_entity_recorder.yaml",
        "//gxf/serialization/tests:test_entity_replayer.yaml",
    ],
    deps = [
        "//gxf/core:gxf",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_file",
    srcs = ["test_file.cpp"],
    data = [
        "//gxf/gxe:manifest",
    ],
    deps = [
        "//gxf/core:libgxf_core",
        "//gxf/serialization:file",
        "//gxf/serialization:file_stream",
        "//gxf/std:allocator",
        "@gtest//:main",
    ],
)

nv_gxf_cc_test_group(
    name = "tests",
    apps = [
        "gxf/serialization/tests/test_std_component_serializer_device.yaml",
        "gxf/serialization/tests/test_std_component_serializer_host.yaml",
        "gxf/serialization/tests/test_std_component_serializer_system.yaml",
    ],
    extensions = [
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/serialization",
        "//gxf/serialization/tests:test_serialization",
    ],
    manifest_path_hack = "gxf/serialization/tests/",
    deps = ["//gxf/test:test_main"],
)

nv_gxf_test_app(
    name = "test_device_serializer",
    srcs = ["test_std_component_serializer_device.yaml"],
    extensions = [
        "//gxf/serialization/tests:test_serialization",
        "//gxf/test/extensions:test",
    ]
)

nv_gxf_cc_extension(
    name = "test_serialization",
    srcs = ["test_serialization.cpp"],
    deps = [
        "serialization_tester",
        "//gxf/std:default_extension",
    ],
    ext_deps = [
        "//gxf/std",
        "//gxf/serialization",
    ],
)

nv_gxf_cc_library(
    name = "serialization_tester",
    srcs = ["serialization_tester.cpp"],
    hdrs = ["serialization_tester.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/serialization:entity_serializer",
        "//gxf/serialization:serialization_buffer",
        "//gxf/serialization:endpoint",
        "//gxf/std",
        "//gxf/core:gxf",
        "//common",
    ],
)
