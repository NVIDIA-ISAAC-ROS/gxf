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
    "nv_gxf_cc_extension",
    "nv_gxf_cc_library",
)


py_library(
    name = "network_extension",
    srcs = [
        "__init__.py",
        "Components.py",
    ],
    visibility = ["//visibility:public"],
    data = [
        "//gxf/network:Components.py",
    ],
)

nv_gxf_cc_extension(
    name = "network",
    srcs = ["network.cpp"],
    deps = [
        "clock_sync_primary",
        "clock_sync_secondary",
        "tcp_codelet",
        "tcp_client",
        "tcp_server",
        "//gxf/core:gxf",
        "//gxf/std:extension_factory_helper",
    ],
    ext_deps = [
        "//gxf/std",
        "//gxf/serialization",
    ],
)

nv_gxf_cc_library(
    name = "clock_sync_primary",
    srcs = ["clock_sync_primary.cpp"],
    hdrs = ["clock_sync_primary.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/std:clock",
        "//gxf/std:codelet",
        "//gxf/std:transmitter",
    ],
)

nv_gxf_cc_library(
    name = "clock_sync_secondary",
    srcs = ["clock_sync_secondary.cpp"],
    hdrs = ["clock_sync_secondary.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/std:codelet",
        "//gxf/std:receiver",
        "//gxf/std:synthetic_clock",
    ],
)

nv_gxf_cc_library(
    name = "tcp_codelet",
    srcs = ["tcp_codelet.cpp"],
    hdrs = ["tcp_codelet.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "tcp_client_socket",
        "tcp_server_socket",
        "//gxf/core:parameter_parser_std",
        "//gxf/serialization:entity_serializer",
        "//gxf/std:codelet",
        "//gxf/std:scheduling_terms",
    ],
)

nv_gxf_cc_library(
    name = "tcp_client",
    srcs = ["tcp_client.cpp"],
    hdrs = ["tcp_client.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "tcp_codelet",
    ],
)

nv_gxf_cc_library(
    name = "tcp_server",
    srcs = ["tcp_server.cpp"],
    hdrs = ["tcp_server.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "tcp_codelet",
    ],
)

nv_gxf_cc_library(
    name = "tcp_server_socket",
    srcs = ["tcp_server_socket.cpp"],
    hdrs = ["tcp_server_socket.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "tcp_client_socket",
    ],
)

nv_gxf_cc_library(
    name = "tcp_client_socket",
    srcs = ["tcp_client_socket.cpp"],
    hdrs = ["tcp_client_socket.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/serialization:endpoint",
        "//gxf/serialization:entity_serializer",
    ],
)

load("//registry/build:registry.bzl", "register_extension")

register_extension(
    name = "register_network_ext",
    badges = [""],
    extension = "network",
    headers = [
        "BUILD.public",
        "tcp_client.hpp",
        "tcp_server.hpp",
    ],
    python_sources = [
        "//gxf/network:__init__.py",
        "//gxf/network:Components.py",
        ],
    labels = ["gxf"],
    license_file = ":LICENSE",
    local_dependencies = [
        "//gxf/std:register_std_ext",
        "//gxf/serialization:register_serialization_ext",
    ],
    priority = "1",
    url = "www.nvidia.com",
    uuid = "f50665e5-ade2-f71b-de2a-2380614b1725",
    version = "1.6.0",
    visibility = ["//visibility:public"],
)
