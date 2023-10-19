"""
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load(
    "@com_nvidia_gxf//gxf:repo.bzl",
    "nv_gxf_http_archive",
    "nv_gxf_new_git_repository",
    "nv_gxf_git_repository",
    "nv_gxf_new_local_repository",
)

def clean_dep(dep):
    return str(Label(dep))

def {{extension_short_name}}_workspace():
    pass