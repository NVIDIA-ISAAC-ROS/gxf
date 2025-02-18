"""
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

_workspace_name = "com_nvidia_{{extension_short_name}}_gxf"
workspace(name = _workspace_name)

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

####################################################################################
# Load gxf core and its dependencies

# Patch remove qnx and nvcc toolchain from GXF toolchain template to improve build perf
git_repository(
    name = "com_nvidia_gxf",
    remote = "ssh://git-master.nvidia.com:12001/gxf/gxf",
    branch = "master",
)

load("@com_nvidia_gxf//gxf:repo.bzl", "nv_gxf_git_repository", "nv_gxf_new_git_repository")
load("@com_nvidia_gxf//third_party:cuda.bzl", "cuda_workspace")
load("@com_nvidia_gxf//third_party:gxf.bzl", "gxf_test_data", "gxf_workspace", "gxf_python_workspace", "gxf_tools_workspace")
load("@com_nvidia_gxf//third_party:nvsci.bzl", "nvsci_workspace")
load("@com_nvidia_gxf//third_party/ucx:ucx.bzl", "ucx_workspace")
load("@com_nvidia_gxf//third_party/ipc:ipc.bzl", "ipc_workspace")

ucx_workspace()
ipc_workspace()
nvsci_workspace()
cuda_workspace()
gxf_workspace()
gxf_python_workspace()
gxf_tools_workspace()
gxf_test_data()

# Configures toolchain
load("@com_nvidia_gxf//engine/build/toolchain:toolchain.bzl", "toolchain_configure")
toolchain_configure(name = "toolchain")

# Load compilation database repo used to support static analysis tools
# Package created from - https://github.com/xiay-nv/bazel-compilation-database.git
# commit = "c058a4f4646849fa11a3c71f0efd35ae776e8f02"
nv_gxf_http_archive(
    name = "compdb",
    sha256 = "279299e9fda917a3793b4541dfa4d4bcff02d88d9e11191bbc79f659cb6cf72c",
    strip_prefix = "bazel-compilation-database-c058a4f4646849fa11a3c71f0efd35ae776e8f02",
    url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/bazel-compilation-database/c058a4f4646849fa11a3c71f0efd35ae776e8f02.tar.gz"
    licenses = [
        "Apache License 2.0",
        "https://github.com/xiay-nv/bazel-compilation-database/blob/c058a4f4/LICENSE",
    ],
    type = "tar.gz",
)


load("@compdb//:config.bzl", "config_clang_compdb")
config_clang_compdb(
    workspace_name = _workspace_name,
)

# Only needed if using the packaging rules.
load("@rules_python//python:pip.bzl", "pip_install")
pip_install(
    name = "pip_dependencies",
    requirements = "@com_nvidia_gxf//registry/build:requirements.txt"
)

#####################################################################################

# {{extension_short_name}} dependencies
load("//third_party:{{extension_short_name}}.bzl", "{{extension_short_name}}_workspace")

{{extension_short_name}}_workspace()
