"""
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//gxf:gxf.bzl", "nv_gxf_app", "nv_gxf_pkg", "nv_gxf_test_app", "nv_gxf_sub_graph")
load("//registry/build:runtime.bzl", "gxf_registry_app")

# Sub graph targets
nv_gxf_sub_graph(
    name = "gather_forward_sub_graph",
    sub_graphs = [
        "forward_subgraph.yaml",
        "gather_subgraph.yaml",
        "gather_subgraph.param.yaml"
    ],
    extensions = [
        "//gxf/test/extensions:test",
    ]
)

nv_gxf_sub_graph(
    name = "forward_sub_graph",
    sub_graphs = [
        "forward_subgraph.yaml",
    ],
    extensions = [
        "//gxf/test/extensions:test",
    ]
)

nv_gxf_sub_graph(
    name = "gather_sub_graph",
    sub_graphs = [
        "gather_subgraph.yaml",
        "gather_subgraph.param.yaml"
    ],
    extensions = [
        "//gxf/test/extensions:test",
    ]
)

#################################################################################
# All test apps below use the gxf registry selecting the required extensions

gxf_registry_app(
    name = "test_ping_registry",
    graphs = [
        "test_ping_composer.yaml",
        "test_ping_composer_parameters.yaml"],
)

gxf_registry_app(
    name = "test_ping_multithread_registry",
    graphs = ["test_ping_multi_thread.yaml"],
)

gxf_registry_app(
    name = "test_subgraph_bazel_target_registry",
    sub_graphs = ["gather_forward_sub_graph"],
    graphs = [
        "test_subgraph_split.yaml",
        "test_subgraph_split.param2.yaml",
    ],
)

gxf_registry_app(
    name = "test_subgraph_bazel_target_multi_registry",
    sub_graphs = ["forward_sub_graph", "gather_sub_graph"],
    graphs = [
        "test_subgraph_split.yaml",
        "test_subgraph_split.param2.yaml",
    ],
)
