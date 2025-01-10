"""
 SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

# Unit test

cc_test(
   name = "test_http_no_yaml",
   srcs = ["test_http_no_yaml.cpp"],
   data = [
       "//gxf/std:libgxf_std.so",
       "//gxf/ipc/http:libgxf_http.so",
       "//gxf/test/extensions:libgxf_test.so",
    ],
   deps = [
       "//gxf/core:gxf",
       "//gxf/ipc/http",
       "//gxf/ipc/http:http_ipc_client",
       "//gxf/ipc/http:http_server",
       "@gtest//:main",
   ],
)

load(
    "@com_nvidia_gxf//gxf:gxf.bzl",
    "nv_gxf_cc_extension",
    "nv_gxf_cc_library",
    "nv_gxf_app"
)
load("@com_nvidia_gxf//packaging:packaging.bzl", "nv_gxf_pkg_extension")
load("@rules_pkg//pkg:mappings.bzl", "pkg_attributes", "pkg_filegroup", "pkg_files", "pkg_mkdirs", "strip_prefix")

nv_gxf_cc_extension(
    name = "test_http",
    srcs = ["test_http.cpp"],
    deps = [
        "mock_http_service",
        "@com_nvidia_gxf//gxf/std:extension_factory_helper",
    ],
    ext_deps = [
        "//gxf/ipc/http",
    ],
)

exports_files(["LICENSE"])

nv_gxf_pkg_extension (
    name = "test_http",
    extension = "test_http",
    build = "BUILD.release",
    license = "LICENSE",
    headers = [
      "mock_http_service.hpp",
    ],
)

nv_gxf_cc_library(
    name = "mock_http_service",
    visibility = ["//visibility:public"],
    srcs = ["mock_http_service.cpp"],
    hdrs = ["mock_http_service.hpp"],
    deps = [
        "//gxf/ipc/http",
    ]
)

nv_gxf_app(
    name = "test_http_server_client",
    extensions = [
        "@com_nvidia_gxf//gxf/std",
        "//gxf/ipc/http",
        "//gxf/ipc/http/tests:test_http",
    ],
    gxf_file = "test_http_server_client.yaml",
)

nv_gxf_app(
    name = "test_http_server_ipc_client",
    extensions = [
        "@com_nvidia_gxf//gxf/std",
        "//gxf/ipc/http",
        "//gxf/ipc/http/tests:test_http",
    ],
    gxf_file = "test_http_server_ipc_client.yaml",
)