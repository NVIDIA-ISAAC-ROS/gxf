"""
 SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cc_library(
    name = "python_x86_64_3_7",
    srcs = [
        "lib/python3.7/config-3.7-x86_64-linux-gnu/libpython3.7m.so.1",
        "lib/python3.7/config-3.7-x86_64-linux-gnu/libpython3.7m.so.1.0",
    ],
    hdrs = glob([
        "include/python3.7/**/*.h",
    ]),
    linkopts = [],
    strip_include_prefix = "include/python3.7",
    visibility = ["//visibility:public"],
    deps = [
        ":python_x86_64_3_7_hdr",
    ],
)

cc_library(
    name ="python_x86_64_3_7_hdr",
    srcs = [],
    hdrs = glob([
        "include/x86_64-linux-gnu/python3.7m/*.h",
    ]),
    linkopts = [],
    strip_include_prefix = "include/",
    deps = [],
)
