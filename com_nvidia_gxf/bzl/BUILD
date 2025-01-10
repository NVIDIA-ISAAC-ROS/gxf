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

load("@io_bazel_rules_docker//container:layer.bzl", "container_layer")

# helper genrule to create a version.txt containing the current git hash, to be used to version Docker containers
genrule(
    name = "version_txt",
    srcs = [],
    outs = ["version.txt"],
    cmd = "(cat bazel-out/stable-status.txt | grep 'STABLE_SCM_HASH_SHORT ' | cut -f2 -d' ') > $@",
    stamp = 1,
)

# a layer containing a single file with the current git hash, this can be pushed on top of app layers to version them
container_layer(
    name = "version_txt_layer",
    files = ["version_txt"],
    visibility = ["//visibility:public"],
)
