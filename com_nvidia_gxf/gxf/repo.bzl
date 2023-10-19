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

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def nv_gxf_http_archive(licenses, name, **kwargs):
    """
    A GXF HTTP third party archive. Augment the standard Bazel HTTP archive workspace rule.
    Mandatory licenses label.
    """
    maybe(
        repo_rule = http_archive,
        name = name,
        **kwargs
    )

def nv_gxf_new_git_repository(licenses, name, **kwargs):
    """
    A GXF Git third party repository. Augment the standard new Bazel Git repository workspace
    rule. Mandatory licenses label.
    """
    maybe(
        repo_rule = new_git_repository,
        name = name,
        **kwargs
    )

def nv_gxf_git_repository(licenses, name, **kwargs):
    """
    A GXF Git third party repository. Augment the standard Bazel Git repository workspace rule.
    Mandatory licenses label.
    """
    maybe(
        repo_rule = git_repository,
        name = name,
        **kwargs
    )

def nv_gxf_new_local_repository(licenses, name, **kwargs):
    """
    A GXF local third party repository. Augment the standard Bazel Git repository workspace rule.
    Mandatory licenses label.
    """
    maybe(
        repo_rule = native.new_local_repository,
        name = name,
        **kwargs
    )

