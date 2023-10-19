"""
Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load(
    "//gxf:gxf.bzl",
    "nv_gxf_app",
)

cc_test(
    name = "test_behavior",
    srcs = ["test_behavior.cpp"],
    data = [
        "test_behavior_tree_constant.yaml",
        "test_behavior_tree_parallel.yaml",
        "test_behavior_tree_parallel_success.yaml",
        "test_behavior_tree_parallel_failure.yaml",
        "test_behavior_tree_sel.yaml",
        "test_behavior_tree_sel_failure.yaml",
        "test_behavior_tree_seq.yaml",
        "test_behavior_tree_seq_failure.yaml",
        "test_behavior_tree_switch.yaml",
        "test_behavior_tree_timer.yaml",
        "test_behavior_tree_epoch.yaml",
        "test_behavior_tree_epoch_sch_with_all_behaviors.yaml",
        "//gxf/gxe:manifest",
    ],
    deps = [
        "//gxf/core:gxf",
        "@gtest//:main",
        "@yaml-cpp",
        "//gxf/std",
    ],
)

nv_gxf_app(
    name = "test_repeat_behavior",
    extensions = [
        "//gxf/std",
        "//gxf/behavior_tree:behavior_tree",
    ],
    gxf_file = "test_behavior_tree_seq_repeat_controller.yaml"
)

nv_gxf_app(
    name = "test_behavior_tree_parallel",
    extensions = [
        "//gxf/std",
        "//gxf/behavior_tree:behavior_tree",
    ],
    gxf_file = "test_behavior_tree_parallel.yaml",
)

nv_gxf_app(
    name = "test_behavior_tree_seq",
    extensions = [
        "//gxf/std",
        "//gxf/behavior_tree:behavior_tree",
    ],
    gxf_file = "test_behavior_tree_seq.yaml",
)

nv_gxf_app(
    name = "test_behavior_tree_switch",
    extensions = [
        "//gxf/std",
        "//gxf/behavior_tree:behavior_tree",
    ],
    gxf_file = "test_behavior_tree_switch.yaml",
)

nv_gxf_app(
    name = "test_behavior_tree_timer",
    extensions = [
        "//gxf/std",
        "//gxf/behavior_tree:behavior_tree",
    ],
    gxf_file = "test_behavior_tree_timer.yaml",
)

nv_gxf_app(
    name = "test_stress_behavior_tree_app",
    extensions = [
        "//gxf/std",
        "//gxf/serialization",
        "//gxf/test/extensions:test",
        "//gxf/behavior_tree:behavior_tree",
    ],
    srcs = ["test_stress_behavior_tree.yaml"],
)

py_binary(
    name = "test_stress_behavior_tree_sel",
    srcs = ["test_stress_behavior_tree_sel.py"],
)

test_stress_behavior_tree_sel_yaml_files = [
    "test_stress_behavior_tree_sel_gen.yaml",
]

genrule(
    name = "test_stress_behavior_tree_sel_gen",
    outs = test_stress_behavior_tree_sel_yaml_files,
    cmd = "$(location :test_stress_behavior_tree_sel) $(RULEDIR)",
    message = "----- Generating the test_stress_behavior_tree_sel_gen.yaml file",
    tools = [":test_stress_behavior_tree_sel"],
    visibility = ["//visibility:public"],
)

nv_gxf_app(
    name = "test_stress_behavior_tree_sel_app",
    extensions = [
        "//gxf/std",
        "//gxf/serialization",
        "//gxf/test/extensions:test",
        "//gxf/behavior_tree:behavior_tree",
    ],
    gxf_file = "test_stress_behavior_tree_sel_gen.yaml",
)

py_binary(
    name = "test_stress_behavior_tree_seq",
    srcs = ["test_stress_behavior_tree_seq.py"],
)

test_stress_behavior_tree_seq_yaml_files = [
    "test_stress_behavior_tree_seq_gen.yaml",
]

genrule(
    name = "test_stress_behavior_tree_seq_gen",
    outs = test_stress_behavior_tree_seq_yaml_files,
    cmd = "$(location :test_stress_behavior_tree_seq) $(RULEDIR)",
    message = "----- Generating the test_stress_behavior_tree_seq_gen.yaml file",
    tools = [":test_stress_behavior_tree_seq"],
    visibility = ["//visibility:public"],
)

nv_gxf_app(
    name = "test_stress_behavior_tree_seq_app",
    extensions = [
        "//gxf/std",
        "//gxf/serialization",
        "//gxf/test/extensions:test",
        "//gxf/behavior_tree:behavior_tree",
    ],
    gxf_file = "test_stress_behavior_tree_seq_gen.yaml",
)
py_binary(
    name = "test_stress_behavior_tree_parallel",
    srcs = ["test_stress_behavior_tree_parallel.py"],
)

test_stress_behavior_tree_parallel_yaml_files = [
    "test_stress_behavior_tree_parallel_gen.yaml",
]

genrule(
    name = "test_stress_behavior_tree_parallel_gen",
    outs = test_stress_behavior_tree_parallel_yaml_files,
    cmd = "$(location :test_stress_behavior_tree_parallel) $(RULEDIR)",
    message = "----- Generating the test_stress_behavior_tree_parallel_gen.yaml file",
    tools = [":test_stress_behavior_tree_parallel"],
    visibility = ["//visibility:public"],
)

nv_gxf_app(
    name = "test_stress_behavior_tree_parallel_app",
    extensions = [
        "//gxf/std",
        "//gxf/serialization",
        "//gxf/test/extensions:test",
        "//gxf/behavior_tree:behavior_tree",
    ],
    gxf_file = "test_stress_behavior_tree_parallel_gen.yaml",
)

py_binary(
    name = "test_stress_behavior_tree_nested",
    srcs = ["test_stress_behavior_tree_nested.py"],
)

test_stress_behavior_tree_nested_yaml_files = [
    "test_stress_behavior_tree_nested_gen.yaml",
]

genrule(
    name = "test_stress_behavior_tree_nested_gen",
    outs = test_stress_behavior_tree_nested_yaml_files,
    cmd = "$(location :test_stress_behavior_tree_nested) $(RULEDIR)",
    message = "----- Generating the test_stress_behavior_tree_nested_gen.yaml file",
    tools = [":test_stress_behavior_tree_nested"],
    visibility = ["//visibility:public"],
)

nv_gxf_app(
    name = "test_stress_behavior_tree_nested_app",
    extensions = [
        "//gxf/std",
        "//gxf/serialization",
        "//gxf/test/extensions:test",
        "//gxf/behavior_tree:behavior_tree",
    ],
    gxf_file = "test_stress_behavior_tree_nested_gen.yaml",
)

nv_gxf_app(
    name = "test_behavior_tree_multi_thread_scheduler_app",
    extensions = [
        "//gxf/std",
        "//gxf/serialization",
        "//gxf/test/extensions:test",
        "//gxf/behavior_tree:behavior_tree",
    ],
    srcs = ["test_behavior_tree_multi_thread_scheduler.yaml"],
)

nv_gxf_app(
    name = "test_behavior_tree_with_non_behavior_tree_codelet_as_child_app",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/serialization",
        "//gxf/test/extensions:test",
        "//gxf/behavior_tree:behavior_tree",
    ],
    srcs = ["test_behavior_tree_with_non_behavior_tree_codelet_as_child.yaml"],
)
