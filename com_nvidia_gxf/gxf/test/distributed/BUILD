"""
 SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

load("//gxf:gxf.bzl", "nv_gxf_app", "nv_gxf_multi_app")

nv_gxf_app(
    name = "test_graph_worker",
    extensions = [
        "//gxf/ipc/grpc",
        "//gxf/multimedia",
        "//gxf/sample",
        "//gxf/serialization",
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/ucx",
    ],
    data = [
        "//gxf/ucx/tests:test_ucx_ping",
    ],
    gxf_file = "test_graph_worker.yaml",
)

nv_gxf_app(
    name = "test_graph_driver",
    extensions = [
        "//gxf/std",
        "//gxf/ipc/grpc",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_graph_driver.yaml",
)

nv_gxf_multi_app(
    name = "test_distributed",
    apps = [
        "test_graph_driver",
        "test_graph_worker",
    ],
    tags = ["exclusive", "manual"],
    visibility = ["//visibility:public"],
)
