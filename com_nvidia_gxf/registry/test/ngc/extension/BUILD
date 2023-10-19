"""
Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load(
    "//gxf:gxf.bzl",
    "nv_gxf_cc_extension",
)

load(
    "//registry/build:registry.bzl",
    "register_extension"
)

load(
    "//registry/build:gxf_extension.bzl",
    "publish_extension"
)

exports_files(["LICENSE"])

nv_gxf_cc_extension(
    name = "ngc_test",
    srcs = ["ngc_test.cpp"],
    deps = [
          "//gxf/std:extension_factory_helper",
          "//gxf/core:gxf"
          ],
    ext_deps = [
        "//gxf/std",
    ],
)

register_extension(
    name = "register_ngc_test_ext",
    extension = "ngc_test",
    uuid = "fb5bcf10-43f3-44bc-846f-d95a0a077045",
    version = "1.1.0",
    license_file = ":LICENSE",
    url = "www.nvidia.com",
    priority = "1",
    headers = ["header.hpp"],
    binaries = ["test_binary.a"],
    visibility = ["//visibility:public"],
    local_dependencies = ["//gxf/std:register_std_ext"],
    compute_dependencies = ["cuda", "cudnn", "tensorrt", "deepstream", "triton","vpi"],
)

publish_extension(
    name = "publish_ngc_test_ext",
    ext_name = "NGCTestExtension",
    ext_registration_target = ":register_ngc_test_ext",
    repo_name = "test1",
    arch = "x86_64",
    distribution = "ubuntu_20.04",
    os = "linux",
    cuda = "12.1",
    cudnn = "8.8.1",
    tensorrt = "8.5.3",
    deepstream = "6.2.1",
    triton = "2.32.0",
    vpi = "2.3.1",
    force = True,
)