"""
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("@com_nvidia_gxf//gxf:gxf.bzl",
        "nv_gxf_cc_extension",
        "nv_gxf_cc_library",
        "nv_gxf_cc_binary"
)
load("@com_github_grpc_grpc//bazel:grpc_build_system.bzl", "grpc_proto_library")
load("@com_nvidia_gxf//packaging:packaging.bzl", "nv_gxf_pkg_extension")

grpc_proto_library(
    name = "grpc_service_proto",
    srcs = ["grpc_service.proto"]
)


nv_gxf_cc_library(
    name = "grpc_client",
    srcs = ["grpc_client.cpp"],
    hdrs = ["grpc_client.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_nvidia_gxf//gxf/std:ipc_client",
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_grpc_grpc//src/proto/grpc/health/v1:health_proto",
        "grpc_service_proto",
    ],
    copts = ["-fvisibility=hidden",
             "-fvisibility-inlines-hidden"],
)

nv_gxf_cc_library(
    name = "grpc_server",
    srcs = ["grpc_server.cpp"],
    hdrs = ["grpc_server.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_nvidia_gxf//gxf/std:ipc_server",
        "@com_github_grpc_grpc//:grpc++",
        "grpc_service_proto"
    ],
    copts = ["-fvisibility=hidden",
             "-fvisibility-inlines-hidden"],
)

nv_gxf_cc_library(
    name = "grpc_libs",
    deps = [
        "grpc_client",
        "grpc_server",
    ],
)

nv_gxf_cc_extension(
    name = "grpc",
    srcs = ["grpc_ext.cpp"],
    deps = [
        "grpc_libs",
        "@com_nvidia_gxf//gxf/core:gxf",
        "@com_nvidia_gxf//gxf/std:extension_factory_helper",
    ],
    ext_deps = [
        "@com_nvidia_gxf//gxf/std",
    ],
)

load("@com_nvidia_gxf//registry/build:registry.bzl", "register_extension")

exports_files(["LICENSE"])

register_extension(
    name = "register_grpc_ext",
    badges = [""],
    extension = "grpc",
    labels = ["gxf"],
    headers = [
        "BUILD.public",
        "grpc_client.hpp",
        "grpc_server.hpp",
    ],
    python_sources = [
        "//gxf/ipc/grpc:Components.py",
        "//gxf/ipc/grpc:__init__.py",
    ],
    license = "NVIDIA",
    license_file = ":LICENSE",
    priority = "1",
    url = "www.nvidia.com",
    uuid = "62e7335c-c555-47c9-a2d2-21991b7fd250",
    version = "0.5.0",
    local_dependencies = ["@com_nvidia_gxf//gxf/std:register_std_ext"],
    visibility = ["//visibility:public"],
)

nv_gxf_pkg_extension(
    name = "grpc",
    extension = "grpc",
    build = "BUILD.release",
    license = "LICENSE",
    headers = [
      "grpc_server.hpp",
      "grpc_client.hpp"
    ],
)
