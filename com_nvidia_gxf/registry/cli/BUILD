"""
Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

package(default_visibility = ["//visibility:public"])
load("@rules_python//python:defs.bzl", "py_binary")

py_library(
  name = "cli_lib",
  deps = [
          "//registry/core:core",
          "//registry/core:core_interface"
          ],
  srcs = [
          "__init__.py",
          "cli_argparse_group_maker.py",
          "cli_pretty_format.py",
          "cli_query_maker.py"
          ],
  srcs_version = "PY3",
  visibility = ["//visibility:public"],
)

py_binary(
  name = "registry_cli",
  deps = [":cli_lib"],
  main = "registry_cli.py",
  srcs = ["registry_cli.py"],
  srcs_version = "PY3",
  visibility = ["//visibility:public"],
)
