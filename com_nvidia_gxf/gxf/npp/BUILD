"""
Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.

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

load("@com_nvidia_gxf//packaging:packaging.bzl", "nv_gxf_pkg_extension")

exports_files([
    "__init__.py",
])

nv_gxf_cc_extension(
    name = "npp",
    srcs = ["npp.cpp"],
    deps = [
        "nppi_mul_c",
        "nppi_set",
    ],
    ext_deps = [
        "//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "nppi_mul_c",
    srcs = ["nppi_mul_c.cpp"],
    hdrs = ["nppi_mul_c.hpp"],
    deps = [
        "//gxf/std",
        "//third_party:nppial",
    ],
)

nv_gxf_cc_library(
    name = "nppi_set",
    srcs = ["nppi_set.cpp"],
    hdrs = ["nppi_set.hpp"],
    deps = [
        "//gxf/std",
        "//third_party:nppidei",
    ],
)

py_library(
    name = "npp_extension",
    srcs = [
        "__init__.py",
        "Components.py"
    ],
    visibility = ["//visibility:public"],
    data = [
        "//gxf/npp:Components.py",
    ],
)


load("//registry/build:registry.bzl", "register_extension")
exports_files(["LICENSE"])

register_extension(
    name = "register_npp_ext",
    extension = "npp",
    uuid = "98839b87-ddb5-4e77-b0df-44c1a5ad136e",
    version = "2.3.0",
    headers = [
        "BUILD.public",
        "nppi_mul_c.hpp",
        "nppi_set.hpp",
    ],
    python_sources = [
        "//gxf/npp:__init__.py",
        "//gxf/npp:Components.py",
    ],
    license_file = ":LICENSE",
    url = "www.nvidia.com",
    labels = ["gxf"],
    badges = [""],
    priority = "1",
    local_dependencies = ["//gxf/std:register_std_ext"],
    visibility = ["//visibility:public"],
)

# extension tar pkg export
nv_gxf_pkg_extension(
    name = "npp",
    extension = "npp",
    build = "BUILD.release",
    license = "LICENSE",
    headers = [
        "nppi_mul_c.hpp",
        "nppi_set.hpp",
    ],
)
