"""
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load(
    "@com_nvidia_gxf//gxf:gxf.bzl",
    "nv_gxf_cc_extension",
    "nv_gxf_cc_library",
)
load("@com_nvidia_gxf//packaging:packaging.bzl", "nv_gxf_pkg_extension")
load("@rules_pkg//pkg:mappings.bzl", "pkg_attributes", "pkg_filegroup", "pkg_files", "pkg_mkdirs", "strip_prefix")

nv_gxf_cc_library(
    name = "http_server",
    srcs = ["http_server.cpp"],
    hdrs = ["http_server.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "@cpprestsdk//:cpprestsdk",
        "@com_nvidia_gxf//gxf/core:gxf_c_api",
        "@com_nvidia_gxf//gxf/core:gxf_cpp_api",
        "@com_nvidia_gxf//gxf/std"
    ]
)

nv_gxf_cc_library(
    name = "http_client",
    srcs = ["http_client_cpprest_impl.cpp"],
    hdrs = [
        "http_client_cpprest_impl.hpp",
        "http_client.hpp"
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@cpprestsdk//:cpprestsdk",
        "@com_nvidia_gxf//gxf/core:gxf_c_api",
        "@com_nvidia_gxf//gxf/core:gxf_cpp_api",
    ]
)

nv_gxf_cc_library(
    name = "http_ipc_client",
    srcs = ["http_ipc_client.cpp"],
    hdrs = ["http_ipc_client.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "@cpprestsdk//:cpprestsdk",
        "@com_nvidia_gxf//gxf/core:gxf_c_api",
        "@com_nvidia_gxf//gxf/core:gxf_cpp_api",
        "@com_nvidia_gxf//gxf/std"
    ]
)

nv_gxf_cc_extension(
    name = "http",
    srcs = ["http.cpp"],
    deps = [
        "http_client",
        "http_ipc_client",
        "http_server",
        "@com_nvidia_gxf//gxf/std:extension_factory_helper",
    ],
    ext_deps = [
        "@com_nvidia_gxf//gxf/std",
    ],
)

load("@com_nvidia_gxf//registry/build:registry.bzl", "register_extension")

exports_files(["LICENSE"])
register_extension(
    name = "register_http_ext",
    badges = [""],
    extension = "http",
    uuid = "22a21c97-f20b-4c03-b277-6e6f52303328",
    version = "0.5.0",
    labels = ["gxf"],
    headers = [
        "BUILD.public",
        "http_client.hpp",
        "http_client_cpprest_impl.hpp",
        "http_ipc_client.hpp",
        "http_server.hpp",
    ],
    python_sources = [
        "//gxf/ipc/http:Components.py",
        "//gxf/ipc/http:__init__.py",
    ],
    license = "NVIDIA",
    license_file = ":LICENSE",
    url = "www.nvidia.com",
    local_dependencies = ["@com_nvidia_gxf//gxf/std:register_std_ext"],
    priority = "1",
    visibility = ["//visibility:public"],
)

nv_gxf_pkg_extension (
    name = "http",
    extension = "http",
    build = "BUILD.release",
    license = "LICENSE",
    headers = [
      "http_client.hpp",
      "http_ipc_client.hpp",
      "http_server.hpp"
    ],
)
