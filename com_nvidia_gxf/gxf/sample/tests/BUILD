"""
 SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

load(
    "//gxf:gxf.bzl",
    "nv_gxf_cc_test_group",
    "nv_gxf_test_app",
)

nv_gxf_cc_test_group(
    name = "sample_test",
    apps = [
        "gxf/sample/tests/test_ping_tx_rx.yaml",
    ],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
    ],
    manifest_path_hack = "gxf/sample/tests/",
    deps = [
        "//gxf/test:test_main",
        "//gxf/std:yaml_file_loader",
        "@yaml-cpp",
    ],
)

nv_gxf_test_app(
    name = "sample_hello_world",
    srcs = ["test_hello_world.yaml"],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
    ],
)