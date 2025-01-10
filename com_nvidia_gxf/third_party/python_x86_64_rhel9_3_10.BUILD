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

cc_library(
    name = "python_x86_64_rhel9_3_10",
    srcs = [
        "lib/python3.10/config-3.10-x86_64-linux-gnu/libpython3.10.so",
    ],
    hdrs = glob([
        "include/python3.10/**/*.h",
    ]),
    linkopts = [],
    strip_include_prefix = "include/python3.10",
    visibility = ["//visibility:public"],
    deps = [],
)
