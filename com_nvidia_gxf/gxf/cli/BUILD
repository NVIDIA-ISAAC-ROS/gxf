"""
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
package(default_visibility = ["//visibility:public"])
load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
  name = "gxf_cli",
  srcs = [
    "__init__.py",
    "gxf_cli.py",
    "http_transport.py",
    "grpc_service_pb2_grpc.py",
    "grpc_service_pb2.py",
    "grpc_transport.py",
    "renderer.py",
    "service_handler.py",
    "transport.py",
    "version.py",
  ],
  srcs_version = "PY3",
  visibility = ["//visibility:public"]
)