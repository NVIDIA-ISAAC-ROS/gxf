"""
 SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

load(
    "//gxf:repo.bzl",
    "nv_gxf_http_archive",
    "nv_gxf_new_git_repository",
    "nv_gxf_new_local_repository",
)

# Uncomment the imports below to download extensions from NGC
# load(
#     "//registry/test/ngc/import:gxf_extension_test.bzl",
#     "import_extensions"
# )

def clean_dep(dep):
    return str(Label(dep))

def gxf_workspace():
    """Loads external dependencies required to build apps with alice"""
    nv_gxf_http_archive(
        name = "yaml-cpp",
        build_file = clean_dep("//third_party:yaml-cpp.BUILD"),
        sha256 = "f38a7a7637993943c4c890e352b1fa3f3bf420535634e9a506d9a21c3890d505",
        url = "https://developer.nvidia.com/isaac/download/third_party/yaml-cpp-0-6-3-tar-gz",
        type = "tar.gz",
        licenses = ["@yaml-cpp//:LICENSE"],
    )

    # JSON for Modern C++
    # version 3.10.5
    # The following library is obtained from : https://github.com/nlohmann/json
    nv_gxf_http_archive(
        name = "nlohmann-json",
        build_file = clean_dep("//third_party:nlohmann_json.BUILD"),
        sha256 = "b94997df68856753b72f0d7a3703b7d484d4745c567f3584ef97c96c25a5798e",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/nlohmann-json-3-10-5.zip",
        type = "zip",
        licenses = ["@nlohmann-json//:LICENSE"],
    )

    nv_gxf_http_archive(
        name = "gtest",
        build_file = clean_dep("//third_party:gtest.BUILD"),
        sha256 = "d88ad7eba129d2d5453da05c6318af6babf65af37835d720e6bfa105d61cf5ce",
        url = "https://developer.nvidia.com/isaac/download/third_party/googletest-release-1-8-0-tar-gz",
        type = "tar.gz",
        strip_prefix = "googletest-release-1.8.0/googletest",
        licenses = ["@gtest//:LICENSE"],
    )

    # breakpad doesn't have version information.
    # following package has the commit id: 13c1568702e8804bc3ebcfbb43 and is present in
    # the master branch here: https://github.com/google/breakpad
    nv_gxf_http_archive(
        name = "breakpad",
        build_file = clean_dep("//third_party:breakpad.BUILD"),
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/external/breakpad-13c1568702e8804bc3ebcfbb435a2786a3e335cf.tar.gz",
        type = "tar.gz",
        sha256 = "9420c263a0db0a0e07a789589f46f0d69b72e921e7eeabb4af3b0018043e5225",
        licenses = ["@breakpad//:LICENSE"],
    )

    # linux-syscall-support doesn't have version information.
    # following package has the commit id: 93426bda6535943ff1525d0460aab5cc0870ccaf and is present
    # in the main branch here:
    # https://chromium.googlesource.com/linux-syscall-support/+/93426bda6535943ff1525d0460aab5cc0870ccaf
    #
    # Patch for gcc 9.3.0 builds
    # https://chromium.googlesource.com/linux-syscall-support/+/8048ece6c16c91acfe0d36d1d3cc0890ab6e945c%5E%21/#F0
    nv_gxf_http_archive(
        name = "lss",
        build_file = clean_dep("//third_party:lss.BUILD"),
        patches = [clean_dep("//third_party:lss_gcc_9_3.patch")],
        sha256 = "6d2e98e9d360797db6348ae725be901c1947e5736d87f07917c2bd835b03eeef",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/external/linux-syscall-support-93426bda6535943ff1525d0460aab5cc0870ccaf.tar.gz",
        type = "tar.gz",
        licenses = ["@lss//:linux_syscall_support.h"],
    )

    native.bind(
        name = "gflags",
        actual = "@com_github_gflags_gflags//:gflags",
    )

    nv_gxf_http_archive(
        name = "com_github_gflags_gflags",
        url = "https://developer.nvidia.com/isaac/download/third_party/gflags-e292e0452fcfd5a8ae055b59052fc041cbab4abf-tar-gz",
        type = "tar.gz",
        sha256 = "a4c5171355e67268b4fd2f31c3f7f2d125683d12e0686fc14893a3ca8c803659",
        licenses = ["@com_github_gflags_gflags//:COPYING.txt"],
    )

    # nvcc-11.6
    # NVCC from https://developer.download.nvidia.com/compute/cuda/11.6.1/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.1-510.47.03-1_amd64.deb
    # Repackaged by nvcc_package_generation.sh
    nv_gxf_http_archive(
        name = "nvcc_11_06",
        build_file = clean_dep("//third_party:nvcc_11_06.BUILD"),
        sha256 = "32cb79d3a4eb6b889c955e208dce78970f3b8bdcf5a0488b952598c0d19b7e80",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/nvcc-11.6-amd64-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    # nvcc-11.7
    # NVCC from http://cuda-repo/release-candidates/kitpicks/cuda-r11-7/11.7.1/014/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.1-515.48.08-1_amd64.deb
    # Repackaged by nvcc_package_generation.sh
    nv_gxf_http_archive(
        name = "nvcc_11_07",
        build_file = clean_dep("//third_party:nvcc_11_07.BUILD"),
        sha256 = "c46c7256947f966f8630dba802486f0047eede06c8f178cef1f4c4fadde736c9",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/nvcc/nvcc-11.7-ubuntu2004-515.48.08-amd64-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    # nvcc-11.8
    # NVCC from http://cuda-internal.nvidia.com/release-candidates/kitpicks/cuda-r11-8/11.8.0/065/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
    # Repackaged by nvcc_package_generation.sh
    nv_gxf_http_archive(
        name = "nvcc_11_08",
        build_file = clean_dep("//third_party:nvcc_11_08.BUILD"),
        sha256 = "d9ed365b7ce220256a6c784568b51fcb310dc63e874f2180f027a3abdda226ac",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic/dependencies/internal/nvcc/nvcc-11.8-ubuntu2004-520.61.05-amd64-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    # nvcc-11.4
    # NVCC from https://developer.download.nvidia.com/compute/cuda/11.4.1/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.1-470.57.02-1_amd64.deb
    # Repackaged by nvcc_package_generation.sh
    nv_gxf_http_archive(
        name = "nvcc_11_04",
        build_file = clean_dep("//third_party:nvcc_11_04.BUILD"),
        sha256 = "8c11a17ff4aba433f6f480ef9ab7535e589afaadd42c87544c174378ffba53d0",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/nvcc/nvcc-11.4.1-ubuntu2004-470.57.02-amd64-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    # nvcc-12.1
    # NVCC from https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.1-530.30.02-1_amd64.deb
    # Repackaged by nvcc_package_generation.sh
    nv_gxf_http_archive(
        name = "nvcc_12_01",
        build_file = clean_dep("//third_party:nvcc_12_01.BUILD"),
        sha256 = "e3ba475f3cb0259f8feb1152cd9b0708f5b68d235bc2f030042537b306944e7d",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/nvcc/nvcc-12.1-amd64-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    # Aarch64 GNU GCC 9.3.0 cross compiler
    #
    # libc.so and pthread.so are linker scripts that include static paths to the root filesystem. The patch modifies the
    # scripts to use relative paths to allow correct linking. Some discussion about the topic and why this becomes an
    # issue with Bazel can be found from:
    # https://stackoverflow.com/questions/52386530/linker-fails-in-sandbox-when-running-through-bazel-but-works-when-sandboxed-comm
    #
    # Toolchain from Jetson L4T Page - https://developer.nvidia.com/embedded/jetson-linux
    #
    nv_gxf_http_archive(
        name = "gcc_9_3_aarch64_linux_gnu",
        build_file = clean_dep("//third_party:gcc_9_3_aarch64_linux_gnu.BUILD"),
        patches = [clean_dep("//third_party:gcc_9_3_aarch64_linux_gnu.patch")],
        url = "https://developer.download.nvidia.com/embedded/L4T/bootlin/aarch64--glibc--stable-final.tar.gz",
        sha256 = "7725b4603193a9d3751d2715ef242bd16ded46b4e0610c83e76d8891cf580975",
        type = "tar.gz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    # composer UI Core:
    nv_gxf_http_archive(
        name = "graph_core",
        sha256 = "b4b614be9a8dd58c1f72a3d80e0147245c32ea28c8fec1ea0e7e8b6a32b1ce7a",
        build_file = "//third_party:graph_editor_core.BUILD",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic/dependencies/internal/omni.exp.graph.core_1.5.2.zip",
        type = "zip",
        licenses = []
    )

    nv_gxf_http_archive(
        name = "rules_pkg",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.6.0/rules_pkg-0.6.0.tar.gz",
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.6.0/rules_pkg-0.6.0.tar.gz",
        ],
        sha256 = "62eeb544ff1ef41d786e329e1536c1d541bb9bcad27ae984d57f18f314018e66",
        type = "tar.gz",
        licenses = ["TBD"],
    )

    # Coverity static analysis
    nv_gxf_http_archive(
        name = "coverity",
        build_file = "@com_nvidia_gxf//coverity/bazel:coverity.BUILD",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/coverity.tar.gz",
        sha256 = "55d7343358289ad89b8d1fe6af80255fb01e1504b09b702f0afe1a914b279b4d",
        type = "tar.gz",
        licenses = ["TBD-Propertietary"],
    )

def gxf_python_workspace():
    """Loads external dependencies required to build gxf python components"""

    nv_gxf_http_archive(
        name = "pybind11",
        build_file = clean_dep("//third_party:pybind11.BUILD"),
        sha256 = "97504db65640570f32d3fdf701c25a340c8643037c3b69aec469c10c93dc8504",
        strip_prefix = "pybind11-2.5.0",
        type = "tar.gz",
        url = "https://developer.nvidia.com/isaac/download/third_party/pybind11-2-5-0-tar-gz",
        licenses = ["TBD"],
    )

    nv_gxf_http_archive(
        name = "python_x86_64",
        build_file = clean_dep("//third_party:python_x86_64.BUILD"),
        sha256 = "54e75648c385f761a774c902cd6d31f016b6eaf27053f51afcad281f6b6fe81a",
        type = "tar.xz",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/python_x86_64_3_8.tar.xz",
        licenses = ["TBD"],
    )

    nv_gxf_http_archive(
        name = "python_x86_64_3_7",
        build_file = clean_dep("//third_party:python_x86_64_3_7.BUILD"),
        sha256 = "2c19434352eddaf0fd684f72d1af5451465bf0e66b81ca2ebe976aff561de571",
        type = "tar.xz",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic/dependencies/internal/python_x86_64_3_7.tar.xz",
        licenses = ["TBD"],
    )

    nv_gxf_http_archive(
        name = "python_aarch64_3_8",
        build_file = clean_dep("//third_party:python_aarch64_3_8.BUILD"),
        sha256 = "e597a529fb5e38817163782e8fdacf61a478e93d045c78ff12dff49d31affe69",
        type = "tar.xz",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/python_3_8_aarch64_jp50.tar.xz",
        licenses = ["TBD"],
    )

    nv_gxf_http_archive(
        name = "rules_python",
        url = "https://github.com/bazelbuild/rules_python/releases/download/0.1.0/rules_python-0.1.0.tar.gz",
        sha256 = "b6d46438523a3ec0f3cead544190ee13223a52f6a6765a29eae7b7cc24cc83a0",
        type = "tar.gz",
        licenses = ["TBD"],
    )

def gxf_tools_workspace():
    """Loads external dependencies required to build/execute tools like registry"""

    gxf_python_workspace()

    nv_gxf_http_archive(
        name = "bazel_skylib",
        url = "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
        type = "tar.gz",
        licenses = ["https://github.com/bazelbuild/bazel-skylib/blob/main/LICENSE"],
    )

def gxf_test_data():
    # import_extensions()
    pass
