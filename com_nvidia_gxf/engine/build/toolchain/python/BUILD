"""
 SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
load("@rules_python//python:defs.bzl", "py_runtime_pair", "py_runtime")

# Set the python version
# py_runtime(
#     name = "py_runtime_x_y",
#     interpreter_path = "/usr/bin/pythonx.y",
#     python_version = "PY3"
# )

py_runtime(
    name = "py_runtime_3_10_ubuntu_22_04",
    interpreter_path = "/usr/bin/python3.10",
    python_version = "PY3"
)

py_runtime(
    name = "py_runtime_3_10_rhel9",
    interpreter_path = "/usr/local/bin/python3.10",
    python_version = "PY3"
)

py_runtime_pair(
    name = "runtime_pair",
    py2_runtime = None,
    py3_runtime = select({
        "//engine/build:ubuntu_python3_10": ":py_runtime_3_10_ubuntu_22_04",
        "//engine/build:rhel_python3_10": ":py_runtime_3_10_rhel9",
        "//conditions:default": ":py_runtime_3_10_ubuntu_22_04",
    }),
)

toolchain(
    name = "py_toolchain",
    toolchain = ":runtime_pair",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type",
)
