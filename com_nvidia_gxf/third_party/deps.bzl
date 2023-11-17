"""
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "patch", "workspace_and_buildfile")

# https://www.appsloveworld.com/cplus/100/170/bazel-relative-local-path-as-url-in-http-archive
# https://bazel.build/rules/lib/repository_ctx#extract
# https://github.com/bazelbuild/bazel/blob/master/tools/build_defs/repo/http.bzl

def _local_archive_impl(repository_ctx):
  repository_ctx.extract(repository_ctx.attr.src, stripPrefix=repository_ctx.attr.strip_prefix)
  patch(repository_ctx)

  if repository_ctx.attr.build_file:
    repository_ctx.file("BUILD.bazel", repository_ctx.read(repository_ctx.attr.build_file))
    workspace_and_buildfile(repository_ctx)

local_archive = repository_rule(
    attrs = {
        "src": attr.label(mandatory = True, allow_single_file = True),
        "strip_prefix": attr.string(),
        "build_file": attr.label(allow_single_file = True),
        "patches": attr.label_list(default = []),
        "patch_args": attr.string_list(default = ["-p0"]),
        "build_file_content": attr.string(doc = "The content for the BUILD file for this repository.",),
        "workspace_file": attr.label(doc = "The file to use as the WORKSPACE file for this repository.",),
        "workspace_file_content": attr.string(doc = "The content for the WORKSPACE file for this repository.",),
    },
    implementation = _local_archive_impl,
)