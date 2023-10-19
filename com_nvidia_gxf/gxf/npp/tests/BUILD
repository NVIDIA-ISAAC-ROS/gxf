"""
 SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
)

nv_gxf_cc_test_group(
    name = "all_test",
    apps = [
        "gxf/npp/tests/test_cuda_tensor_scale.yaml",
    ],
    extensions = [
        "//gxf/std",
        "//gxf/npp",
        "//gxf/test/extensions:test",
    ],
    manifest_path_hack = "gxf/npp/tests/",
    deps = [
        "//gxf/test:test_main",
        "//gxf/std:yaml_file_loader",
        "@yaml-cpp",
    ],
)
