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

load("@com_extension_dev//build/toolchain:toolchain.bzl", "toolchain_configure")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def load_extension_dev_workspace():
    toolchain_configure(name = "toolchain")

    new_git_repository(
        name = "yaml-cpp",
        build_file = "@com_extension_dev//build:third_party/yaml-cpp.BUILD",
        commit = "9a3624205e8774953ef18f57067b3426c1c5ada6",
        remote = "https://github.com/jbeder/yaml-cpp.git",
        shallow_since = "1569430560 -0700",
    )

    http_archive(
        name = "rules_python",
        url = "https://github.com/bazelbuild/rules_python/releases/download/0.1.0/rules_python-0.1.0.tar.gz",
        sha256 = "b6d46438523a3ec0f3cead544190ee13223a52f6a6765a29eae7b7cc24cc83a0",
    )

def graph_cc_extension(
        name,
        interfaces = [],
        srcs = [],
        hdrs = [],
        visibility = None,
        deps = [],
        linkopts = [],
        **kwargs):
    """
    Creates an extension as DSO with file name libgxf_XXX.so
    where XXX is the desired name of the module.
    """

    native.cc_binary(
        name = "libgxf_" + name + ".so",
        srcs = srcs + hdrs,
        visibility = visibility,
        deps = deps + ["@com_extension_dev//:extension_dev"],
        linkopts = linkopts + ["-Wl,-no-undefined"],
        linkshared = True,
        **kwargs
    )

def _graph_nvidia_extension_impl(repository_ctx):
    if not repository_ctx.which("registry"):
        fail("registry not found. Graph Composer tools must be installed")

    ext_uuid = None

    exec_res = repository_ctx.execute([
        "registry",
        "extn",
        "info",
        "-n",
        repository_ctx.name,
    ])
    if exec_res.return_code != 0:
        fail("Failed to get information for extension " + repository_ctx.name +
             ":\n" + exec_res.stderr)

    for line in exec_res.stdout.splitlines():
        if "uuid" in line:
            ext_uuid = line.split(" ")[-1]

    if ext_uuid == None:
        fail("Failed to get information for extension " + repository_ctx.name +
             ":\n" + exec_res.stderr)

    exec_res = repository_ctx.execute([
        "registry",
        "extn",
        "dependencies",
        "-n",
        repository_ctx.name,
        "-s",
        repository_ctx.attr.version,
    ])
    if exec_res.return_code != 0:
        fail("Failed to get dependencies for extension " + repository_ctx.name +
             ":\n" + exec_res.stderr)

    missing_deps = {}
    for line in exec_res.stdout.splitlines():
        if "- name:" in line:
            dep_ext_name = line.split(" ")[3]
            dep_ext_ver = line.split(" ")[5]
            exec_res = repository_ctx.execute([
                "ls",
                repository_ctx.path("../@" + dep_ext_name + ".marker"),
            ])
            if exec_res.return_code != 0:
                missing_deps[dep_ext_name] = dep_ext_ver

    if missing_deps:
        print(" ", "**** Following Nvidia  extensions should be added to workspace if not already ****\n" +
                   "\n".join([n + " version " + v for n, v in missing_deps.items()]))

    exec_res = repository_ctx.execute([
        "registry",
        "extn",
        "import",
        "interface",
        "-n",
        repository_ctx.name,
        "-s",
        repository_ctx.attr.version,
        "-d",
        ".",
    ])

    if exec_res.return_code != 0:
        fail("Failed to import NGC extension " + repository_ctx.name +
             ":\n" + exec_res.stderr)

    exec_res = repository_ctx.execute(["ls", ext_uuid + "/"])
    for f in exec_res.stdout.splitlines():
        if "BUILD" in f:
            repository_ctx.execute(["mv", ext_uuid + "/" + f, "BUILD"])
        else:
            repository_ctx.execute(["mv", ext_uuid + "/" + f, f])

graph_nvidia_extension = repository_rule(
    implementation = _graph_nvidia_extension_impl,
    attrs = {"version": attr.string(mandatory = True)},
)
