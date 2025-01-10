"""
 SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

load("@com_nvidia_gxf//gxf:gxf.bzl", "nv_gxf_app", "nv_gxf_pkg")
load("@com_nvidia_gxf//packaging:packaging.bzl", "nv_gxf_pkg_filegroup")

nv_gxf_app(
    name = "test_ping_statistics",
    extensions = [
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/ipc/grpc",
        "@com_nvidia_gxf//gxf/ipc/http",
        "@com_nvidia_gxf//gxf/test/extensions:test",
    ],
    gxf_file = "test_ping_statistics.yaml",
)

nv_gxf_app(
    name = "test_runtime_config",
    extensions = [
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/ipc/grpc",
        "@com_nvidia_gxf//gxf/ipc/http",
        "@com_nvidia_gxf//gxf/test/extensions:test",
    ],
    gxf_file = "test_runtime_config.yaml",
)

nv_gxf_pkg_filegroup(
    name = "test_apps",
    files =  glob(["*.yaml"]),
)
