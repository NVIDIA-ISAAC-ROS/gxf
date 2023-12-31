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

py_runtime(
    name = "py_runtime_3_7",
    interpreter_path = "/usr/bin/python3.7",
    python_version = "PY3"
)

py_runtime(
    name = "py_runtime_3_8",
    interpreter_path = "/usr/bin/python3.8",
    python_version = "PY3"
)

py_runtime_pair(
    name = "runtime_pair",
    py2_runtime = None,
    py3_runtime = select({
        "//engine/build:python3_7": ":py_runtime_3_7",
        "//conditions:default": ":py_runtime_3_8",
    }),
)

toolchain(
    name = "py_toolchain",
    toolchain = ":runtime_pair",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type",
)
