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

# AARCH64 Python3.8 binary and headers pack grabbed as-is from Jetpack 5.0
cc_library(
    name = "python_aarch64_3_8",
    srcs = [
        "lib/python3.8/config-3.8-aarch64-linux-gnu/libpython3.8.so",
    ],
    hdrs = glob([
        "include/python3.8/**/*.h",
    ]),
    linkopts = [],
    strip_include_prefix = "include//python3.8",
    visibility = ["//visibility:public"],
    deps = [
        ":python_aarch64_3_8_hdr",
    ],
)

cc_library(
    name ="python_aarch64_3_8_hdr",
    srcs = [],
    hdrs = glob([
        "include/aarch64-linux-gnu/python3.8/*.h",
    ]),
    linkopts = [],
    strip_include_prefix = "include/",
    deps = [],
)
