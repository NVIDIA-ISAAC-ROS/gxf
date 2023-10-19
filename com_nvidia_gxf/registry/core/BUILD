"""
Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

package(default_visibility = ["//visibility:public"])
load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_python//python:defs.bzl", "py_library")
load("@pip_dependencies//:requirements.bzl", "requirement")

py_library(
  name = "core",
  deps = [
    "//registry/bindings:select_pygxf",
    requirement("requests"),
    requirement("result"),
    requirement("packaging"),
    requirement("pyyaml"),
    requirement("toml")
  ],
  srcs = [
          "__init__.py",
          "component.py",
          "config.py",
          "core.py",
          "database.py",
          "dependency_governer.py",
          "extension.py",
          "ngc_client.py",
          "packager.py",
          "parameter.py",
          "repository.py",
          "ngc_repository.py",
          "repository_manager.py",
          "utils.py",
          "yaml_loader.py",
          "logger.py",
          "version.py"
          ],
  srcs_version = "PY3",
  visibility = ["//visibility:public"],
)

py_library(
  name = "core_interface",
  srcs = ["core_interface.py"],
  deps = [":core"],
  srcs_version = "PY3",
  visibility = ["//visibility:public"],
)

py_binary(
  name = "dependency_manager",
  srcs = ["dependency_manager.py",
          "logger.py",
          "yaml_loader.py"],
  main = "dependency_manager.py",
  srcs_version = "PY3",
  visibility = ["//visibility:public"],
)
