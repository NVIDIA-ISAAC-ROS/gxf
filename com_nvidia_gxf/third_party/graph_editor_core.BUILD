"""
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("@rules_python//python:defs.bzl", "py_library")

filegroup(
    name = "extension",
    # to locate the extension
    srcs = ["omni.exp.graph.core/config/extension.toml"],
    data = ["graph_widget_core"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "graph_widget_core",
    srcs = [
        "omni.exp.graph.core/omni/exp/graph/core/__init__.py",
        "omni.exp.graph.core/omni/exp/graph/core/builder.py",
        "omni.exp.graph.core/omni/exp/graph/core/commands.py",
        "omni.exp.graph.core/omni/exp/graph/core/exceptions.py",
        "omni.exp.graph.core/omni/exp/graph/core/file_browser.py",
        "omni.exp.graph.core/omni/exp/graph/core/graph_model_proxy.py",
        "omni.exp.graph.core/omni/exp/graph/core/graph_widget_menu.py",
        "omni.exp.graph.core/omni/exp/graph/core/graph_widget.py",
        "omni.exp.graph.core/omni/exp/graph/core/graph_widget_toolbar.py",
        "omni.exp.graph.core/omni/exp/graph/core/graph_window.py",
        "omni.exp.graph.core/omni/exp/graph/core/progress_popup.py",
        "omni.exp.graph.core/omni/exp/graph/core/property_window.py"
    ],
    data = [
      "omni.exp.graph.core/config/extension.toml",
    ] + glob(["icons/**"]),
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)