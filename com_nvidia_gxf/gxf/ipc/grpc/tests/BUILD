"""
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

# Update unit test then disable as it was disabled when first add and never worked
# because of grpc issue. Add nv_gxf_app() at bottom as grpc tests
#
# # Unit test
# cc_test(
#    name = "test_grpc",
#    srcs = ["test_grpc.cpp"],
#    data = [
#        "//gxf/ipc/grpc/tests:test_grpc_server.yaml",
#        "//gxf/std:libgxf_std.so",
#        "//gxf/ipc/grpc:libgxf_grpc.so",
#        "//gxf/test/extensions:libgxf_test.so",
#     ],
#    deps = [
#        "//gxf/core:gxf",
#        "//gxf/ipc/grpc:grpc_client",
#        "//gxf/ipc/grpc:grpc_server",
#        "@com_github_grpc_grpc//:grpc++",
#        "@gtest//:main",
#    ],
# )

# cc_test(
#    name = "test_grpc_no_yaml",
#    srcs = ["test_grpc_no_yaml.cpp"],
#    data = [
#        "//gxf/std:libgxf_std.so",
#        "//gxf/ipc/grpc:libgxf_grpc.so",
#        "//gxf/test/extensions:libgxf_test.so",
#     ],
#    deps = [
#        "//gxf/core:gxf",
#        "//gxf/ipc/grpc",
#        "//gxf/ipc/grpc:grpc_client",
#        "//gxf/ipc/grpc:grpc_server",
#        "@com_github_grpc_grpc//:grpc++",
#        "@gtest//:main",
#    ],
# )

#
# App test
#
load(
    "@com_nvidia_gxf//gxf:gxf.bzl",
    "nv_gxf_cc_extension",
    "nv_gxf_cc_library",
    "nv_gxf_app"
)
load("@com_nvidia_gxf//packaging:packaging.bzl", "nv_gxf_pkg_extension")
load("@rules_pkg//pkg:mappings.bzl", "pkg_attributes", "pkg_filegroup", "pkg_files", "pkg_mkdirs", "strip_prefix")

nv_gxf_cc_extension(
    name = "test_grpc_ext",
    srcs = ["mock_ipc_service_user_ext.cpp"],
    deps = [
        "mock_ipc_service_user",
        "@com_nvidia_gxf//gxf/std:extension_factory_helper",
    ],
    ext_deps = [],
)

exports_files(["LICENSE"])

nv_gxf_pkg_extension (
    name = "test_grpc_ext",
    extension = "test_grpc_ext",
    build = "BUILD.release",
    license = "LICENSE",
    headers = [
      "mock_ipc_service_user.hpp",
    ],
)

nv_gxf_cc_library(
    name = "mock_ipc_service_user",
    visibility = ["//visibility:public"],
    srcs = ["mock_ipc_service_user.cpp"],
    hdrs = ["mock_ipc_service_user.hpp"],
    deps = [
        "//gxf/ipc/grpc",
        "//gxf/std",
    ]
)

nv_gxf_app(
    name = "test_grpc_server_client",
    extensions = [
        "//gxf/std",
        "//gxf/ipc/grpc",
        "//gxf/ipc/grpc/tests:test_grpc_ext",
    ],
    gxf_file = "test_grpc_server_client.yaml",
)