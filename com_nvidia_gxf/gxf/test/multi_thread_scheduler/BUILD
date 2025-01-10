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

load("//gxf:gxf.bzl", "nv_gxf_app", "nv_gxf_pkg", "nv_gxf_test_app", "nv_gxf_sub_graph")

nv_gxf_test_app(
    name = "test_ping_multi_thread",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    srcs = ["test_ping_multi_thread.yaml"],
)

nv_gxf_test_app(
    name = "test_ping_periodic_multi_thread",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    srcs = ["test_ping_periodic_multi_thread.yaml"],
)

nv_gxf_test_app(
    name = "test_periodic_multi_thread",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    srcs = ["test_periodic_multi_thread.yaml"],
)
