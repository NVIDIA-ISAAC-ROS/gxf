"""
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("@com_nvidia_gxf//gxf:gxf.bzl", "nv_gxf_cc_extension", "nv_gxf_cc_library")

nv_gxf_cc_extension(
    name = "{{extension_short_name}}",
    srcs = ["{{extension_short_name}}.cpp"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_nvidia_gxf//gxf/std",
    ],
)

load("@com_nvidia_gxf//registry/build:registry.bzl", "register_extension")

exports_files(["LICENSE"])

register_extension(
    name = "register_{{extension_short_name}}_ext",
    extension = "{{extension_short_name}}",
    labels = ["gxf"],
    badges = [""],
    license_file = ":LICENSE",
    priority = "1",
    url = "www.nvidia.com",
    uuid = "{{full_uuid}}",
    version = "0.0.1",
    local_dependencies = ["@com_nvidia_gxf//gxf/std:register_std_ext"],
    visibility = ["//visibility:public"],
)
