"""
 SPDX-FileCopyrightText: Copyright (c) 2019-2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "cpplint_binary",
    srcs = ["cpplint.py"],
    imports = ["cpplint"],
    main = "cpplint.py",
    visibility = [],
)

alias(
    name = "cpplint",
    actual = ":cpplint_binary",
)

py_binary(
    name = "line_count_collector_binary",
    srcs = ["line_count_collector.py"],
    imports = ["line_count_collector"],
    main = "line_count_collector.py",
    visibility = [],
)

alias(
    name = "line_count_collector",
    actual = ":line_count_collector_binary",
)
