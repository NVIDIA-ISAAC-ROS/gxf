"""
 SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    "nv_gxf_git_repository",
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

    # DLPack for tensor interoperability
    # version 0.8
    # The following library is obtained from : https://github.com/dmlc/dlpack/archive/refs/tags/v0.8.tar.gz
    nv_gxf_http_archive(
        name = "dlpack",
        build_file = clean_dep("//third_party:dlpack.BUILD"),
        sha256 = "cf965c26a5430ba4cc53d61963f288edddcd77443aa4c85ce722aaf1e2f29513",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/dlpack/v0.8.tar.gz",
        type = "tar.gz",
        strip_prefix = "dlpack-0.8",
        licenses = ["@dlpack//:LICENSE"],
    )

    nv_gxf_http_archive(
        name = "gtest",
        build_file = clean_dep("//third_party:gtest.BUILD"),
        sha256 = "ad7fdba11ea011c1d925b3289cf4af2c66a352e18d4c7264392fead75e919363",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/external/googletest-v1.13.0.tar.gz",
        type = "tar.gz",
        strip_prefix = "googletest-1.13.0/googletest",
        licenses = ["@gtest//:LICENSE"],
    )

    # breakpad doesn't have version information.
    # following package has the commit id: bae713be2e51faa5cbe0ac4bcd21c0a3ee72ff8e and is present in
    # the master branch here: https://github.com/google/breakpad
    nv_gxf_http_archive(
        name = "breakpad",
        build_file = clean_dep("//third_party:breakpad.BUILD"),
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/external/breakpad-v2023.01.27.tar.gz",
        type = "tar.gz",
        strip_prefix = "breakpad-2023.01.27",
        sha256 = "f187e8c203bd506689ce4b32596ba821e1e2f034a83b8e07c2c635db4de3cc0b",
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
        patches = [clean_dep("//third_party:lss_gcc.patch")],
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

    # nvcc-12.2
    # NVCC from http://cuda-internal.nvidia.com/release-candidates/kitpicks/cuda-r12-2/12.2.0/039/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
    # Repackaged by nvcc_package_generation.sh
    nv_gxf_http_archive(
        name = "nvcc_12_02",
        build_file = clean_dep("//third_party:nvcc_12_02.BUILD"),
        sha256 = "98a65e57b0ba90ac6a23c93d3ff8a6a495c50e8cf5f2e66604b50e93d27e9855",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/nvcc/nvcc-12.2-amd64-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    # nvcc-12.2
    # NVCC from https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-tegra-repo-ubuntu2204-12-6-local_12.6.0-1_arm64.deb
    nv_gxf_http_archive(
        name = "nvcc_12_06",
        build_file = clean_dep("//third_party:nvcc_12_06.BUILD"),
        sha256 = "347085ba3b3ca4573526dfd344c9f7bab2ee3f16cda26162615da42edaa64ba9",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/nvcc/nvcc-12.6-amd64-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    # Aarch64 GNU GCC 11.3.0 cross compiler
    # //sw/mobile/tools/linux/bootlin/aarch64--glibc--stable-2022.08-1/...
    nv_gxf_http_archive(
        name = "gcc_11_3_aarch64_linux_gnu",
        build_file = clean_dep("//third_party:gcc_11_3_aarch64_linux_gnu.BUILD"),
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/aarch64--gcc_11_3-glibc--stable-2022.08-1.xz",
        sha256 = "1554d0533766a830ad67daeb2b4544ba07ccbf68fd58345b6a9d4a5d8ad292e8",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    # composer UI Core:
    nv_gxf_http_archive(
        name = "graph_core",
        sha256 = "21e2518fdc2475485ecba8a49d29a59f0797f7566baf3d9d25f6e538d20230f0",
        build_file = clean_dep("//third_party:graph_editor_core.BUILD"),
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic/dependencies/internal/omni.exp.graph.core_1.5.3.zip",
        type = "zip",
        licenses = []
    )

    # Package created from - https://github.com/bazelbuild/rules_pkg
    # version - 0.6.0
    nv_gxf_http_archive(
        name = "rules_pkg",
        urls = [
            "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/bazelbuild/rules_pkg/github-rules_pkg-0.6.0.tar.gz",
        ],
        sha256 = "62eeb544ff1ef41d786e329e1536c1d541bb9bcad27ae984d57f18f314018e66",
        type = "tar.gz",
        licenses = ["TBD"],
    )

    # Coverity static analysis
    # Created from //sw/p4/tools/Coverity/2022.12.0
    nv_gxf_http_archive(
        name = "coverity",
        build_file = "@com_nvidia_gxf//coverity/bazel:coverity.BUILD",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/coverity/coverity-2022.12.0.tar.xz",
        sha256 = "05500d8d0c77db0eb06de7914c36edbe8010d5d66dd924abe7fe48c96d5aa22e",
        type = "tar.xz",
        licenses = ["TBD-Propertietary"],
    )

    # Package created from - https://github.com/Neargye/magic_enum
    # version - 0.9.3
    nv_gxf_http_archive(
        name = "magic_enum",
        licenses = ["@magic_enum//:LICENSE"],
        strip_prefix = "magic_enum-0.9.3",
        urls = ["https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/magic_enum/v0.9.3.zip"],
    )

    # Nvidia Rapids RMM library
    # Github repo - https://github.com/rapidsai/rmm/
    # Package archived from - https://github.com/rapidsai/rmm/archive/refs/tags/v24.04.00.tar.gz
    nv_gxf_http_archive(
        name = "rmm",
        build_file = clean_dep("@com_nvidia_gxf//third_party:rmm.BUILD"),
        sha256 = "bb20877c8d92b322dbcb348c2009040784189d3d3c48f93011e13c1b34f6a22f",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/rmm/v24.04.00.tar.gz",
        patches = [clean_dep("@com_nvidia_gxf//third_party:rmm.patch")],
        type = "tar.gz",
        strip_prefix = "rmm-24.04.00",
        licenses = ["TBD"],
    )

    # A modern formatting library (https://fmt.dev/)
    # Github repo - https://github.com/fmtlib/fmt
    # Package archived from - https://github.com/fmtlib/fmt/archive/refs/tags/10.2.1.tar.gz
    nv_gxf_http_archive(
        name = "fmt",
        strip_prefix = "fmt-10.2.1",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/fmt/10.2.1.tar.gz",
        patch_cmds = [
            "mv support/bazel/.bazelversion .bazelversion",
            "mv support/bazel/BUILD.bazel BUILD.bazel",
            "mv support/bazel/WORKSPACE.bazel WORKSPACE.bazel",
        ],
        # Windows-related patch commands are only needed in the case MSYS2 is not installed.
        # More details about the installation process of MSYS2 on Windows systems can be found here:
        # https://docs.bazel.build/versions/main/install-windows.html#installing-compilers-and-language-runtimes
        # Even if MSYS2 is installed the Windows related patch commands can still be used.
        patch_cmds_win = [
            "Move-Item -Path support/bazel/.bazelversion -Destination .bazelversion",
            "Move-Item -Path support/bazel/BUILD.bazel -Destination BUILD.bazel",
            "Move-Item -Path support/bazel/WORKSPACE.bazel -Destination WORKSPACE.bazel",
        ],
        licenses = ["TBD"],
    )

    # Fast C++ logging library
    # Github repo - https://github.com/gabime/spdlog
    # Package archived from - https://github.com/gabime/spdlog/archive/refs/tags/v1.14.1.tar.gz
    nv_gxf_http_archive(
        name = "spdlog",
        build_file = clean_dep("@com_nvidia_gxf//third_party:spdlog.BUILD"),
        sha256 = "1586508029a7d0670dfcb2d97575dcdc242d3868a259742b69f100801ab4e16b",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/spdlog/v1.14.1.tar.gz",
        type = "tar.gz",
        strip_prefix = "spdlog-1.14.1",
        licenses = ["TBD"],
    )

    # Gazelle is a Bazel build file generator for Bazel projects.
    # Github - https://github.com/bazelbuild/bazel-gazelle
    # Package archived from - https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.24.0/bazel-gazelle-v0.24.0.tar.gz
    nv_gxf_http_archive(
        name = "bazel_gazelle",
        licenses = ["@bazel_gazelle//:LICENSE"],
        sha256 = "de69a09dc70417580aabf20a28619bb3ef60d038470c7cf8442fafcf627c21cb",
        urls = ["https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/bazelbuild/bazel-gazelle-v0.24.0.tar.gz"],
    )

    # Go rules for Bazel
    # Github - https://github.com/bazelbuild/rules_go
    # Package archived from - https://github.com/bazelbuild/rules_go/releases/download/v0.24.2/rules_go-v0.24.2.tar.gz
    nv_gxf_http_archive(
        name = "io_bazel_rules_go",
        sha256 = "08c3cd71857d58af3cda759112437d9e63339ac9c6e0042add43f4d94caf632d",
        licenses = ["@io_bazel_rules_go//:LICENSE"],
        urls = ["https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/bazelbuild/rules_go-v0.24.2.tar.gz"],
    )

    # Rules for building and handling Docker images with Bazel
    # Github - https://github.com/bazelbuild/rules_docker
    # Package archived from - https://github.com/bazelbuild/rules_docker/releases/download/v0.22.0/rules_docker-v0.22.0.tar.gz
    nv_gxf_http_archive(
        name = "io_bazel_rules_docker",
        licenses = ["@io_bazel_rules_docker//:LICENSE"],
        patches = ["@com_nvidia_gxf//third_party:rules_docker.context_dir.patch"],
        sha256 = "59536e6ae64359b716ba9c46c39183403b01eabfbd57578e84398b4829ca499a",
        strip_prefix = "rules_docker-0.22.0",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/bazelbuild/rules_docker-v0.22.0.tar.gz",
    )

def gxf_python_workspace():
    """Loads external dependencies required to build gxf python components"""

    nv_gxf_http_archive(
        name = "pybind11",
        build_file = clean_dep("//third_party:pybind11.BUILD"),
        sha256 = "4744701624538da603dde2b533c5a56fac778ea4773650332fe6701b25f191aa",
        strip_prefix = "pybind11-2.11.1",
        type = "tar.xz",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/external/pybind11-2.11.1.tar.gz",
        licenses = ["TBD"],
    )

    nv_gxf_http_archive(
        name = "python_x86_64_3_10",
        build_file = clean_dep("//third_party:python_x86_64_3_10.BUILD"),
        sha256 = "f074a1496d6976ca68b90fb0d69d362f9afa50169b2e580bfeeaa4f1be6a69b2",
        type = "tar.xz",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic/dependencies/internal/python_x86_64_3_10.tar.xz",
        licenses = ["TBD"],
    )

    nv_gxf_http_archive(
        name = "python_x86_64_rhel9_3_10",
        build_file = clean_dep("//third_party:python_x86_64_rhel9_3_10.BUILD"),
        sha256 = "f5cd22dcdfe444a018234ec82d73b98adb8475723f1673ca2d513547dc9c2a44",
        type = "tar.xz",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/python/python_rhel9_x86_64_3_10.tar.xz",
        licenses = ["TBD"],
    )

    nv_gxf_http_archive(
        name = "python_aarch64_3_10",
        build_file = clean_dep("//third_party:python_aarch64_3_10.BUILD"),
        sha256 = "4b79163e2b0ca71f663a1f9dd653e07ec0a17208e9ba56bd7aa54adb5cf60b67",
        type = "tar.xz",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/python_3_10_aarch64_jp60.tar.xz",
        licenses = ["TBD"],
    )

    nv_gxf_http_archive(
        name = "rules_python",
        sha256 = "5fa3c738d33acca3b97622a13a741129f67ef43f5fdfcec63b29374cc0574c29",
        strip_prefix = "rules_python-0.9.0",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/bazelbuild/rules_python/0.9.0.tar.gz",
        type = "tar.gz",
        licenses = ["TBD"],
    )

def gxf_tools_workspace():
    """Loads external dependencies required to build/execute tools like registry"""

    gxf_python_workspace()

    nv_gxf_http_archive(
        name = "bazel_skylib",
        sha256 = "f7be3474d42aae265405a592bb7da8e171919d74c16f082a5457840f06054728",
        urls = [
            "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/external/bazel-skylib-1.2.1.tar.gz",
            "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/bazelbuild/bazel-skylib/bazel-skylib-1.2.1.tar.gz",
        ],
        type = "tar.gz",
        licenses = ["https://github.com/bazelbuild/bazel-skylib/blob/main/LICENSE"],
    )

def gxf_test_data():
    # import_extensions()
    pass
