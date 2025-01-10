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
    )

CUDA_SO = [
    "cudart",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nvgraph",
    "nvrtc",
]

NPP_SO = [
    "nppc",
    "nppial",
    "nppicc",
    "nppicom",
    "nppidei",
    "nppif",
    "nppig",
    "nppim",
    "nppist",
    "nppisu",
    "nppitc",
    "npps",
]

NPP_STATIC = [
    "nppidei",
    "nppial",
    "nppicc",
    "nppicom",
    "nppif",
    "nppig",
    "nppim",
    "nppist",
    "nppisu",
    "nppitc",
]

# Get the path for the shared library with given name for the given version
def cuda_so_path(name, version):
    major_version = version.split(".")[0]
    return "usr/local/cuda-" + version + "/lib64/lib" + name + ".so*"

# Get the path for the static library with given name for the given version
def cuda_static_lib_path(name, version):
    major_version = version.split(".")[0]
    return "usr/local/cuda-" + version + "/lib64/lib" + name + "_static.a"

# Get the path for libcuda.so for the given version. A stub is used as the library is provided
# by the CUDA driver and is required to be available on the system.
def cuda_driver_so_path(family, version):
    if family == "aarch64-qnx":
        return "usr/local/cuda-" + version + "/targets/" + family + "/lib/stubs/libcuda.so"
    elif family == 'aarch64_sbsa':
        return "usr/local/cuda-" + version + "/targets/" + "sbsa-linux" + "/lib/stubs/libcuda.so"
    else:
        return "usr/local/cuda-" + version + "/targets/" + family + "-linux/lib/stubs/libcuda.so"

# Get the path for libnvToolsExt.so for the given version. A stub is used as the library is provided
# by the CUDA driver and is required to be available on the system.
def cuda_nv_tools_ext_so_path(family, version):
    if family == "aarch64-qnx":
        return "usr/local/cuda-" + version + "/targets/" + family + "/lib/libnvToolsExt.so.1"
    elif family == 'aarch64_sbsa':
        return "usr/local/cuda-" + version + "/targets/" + "sbsa-linux" + "/lib/libnvToolsExt.so.1"
    else:
        return "usr/local/cuda-" + version + "/targets/" + family + "-linux/lib/libnvToolsExt.so.1"

# Creates CUDA related dependencies. The arguments `family` and `version` are used to find the
# library and header files in the package
def cuda_device_deps(family, version):
    cuda_target_prefix = "usr/local/cuda-" + version
    cuda_include_prefix = cuda_target_prefix + "/include"
    # CUDA
    cuda_hdrs = native.glob([
        # FIXME separate out headers
        cuda_include_prefix + "/*.h",
        cuda_include_prefix + "/*.hpp",
        cuda_include_prefix + "/CL/*.h",
        cuda_include_prefix + "/crt/*",
        cuda_include_prefix + "/nvtx3/**/*.h",
        cuda_include_prefix + "/cuda/std/**/*",
        cuda_include_prefix + "/cuda/std/detail/libcxx/**/*",
        cuda_include_prefix + "/cuda/**/*",
        cuda_include_prefix + "/thrust/**/*",
        cuda_include_prefix + "/cub/**/*",
        cuda_include_prefix + "/nv/**/*",
    ])

    native.cc_library(
        name = "cuda_headers",
        hdrs = cuda_hdrs,
        strip_include_prefix = cuda_include_prefix,
        visibility = ["//visibility:public"],
    )

    native.cc_library(
        name = "cuda",
        hdrs = cuda_hdrs,
        srcs = [cuda_driver_so_path(family, version), cuda_nv_tools_ext_so_path(family, version)],
        strip_include_prefix = cuda_include_prefix,
        visibility = ["//visibility:public"],
    )

    # Header only library
    native.cc_library(
        name = "nvtx",
        hdrs = native.glob([
                cuda_include_prefix + "/nvtx3/*",
                cuda_include_prefix + "/nvtx3/nvtxDetail/*",
                ]),
        strip_include_prefix = cuda_include_prefix,
        visibility = ["//visibility:public"],
    )

    # Create one library per CUDA shared libray
    for so in CUDA_SO:
        native.cc_library(
            name = so,
            hdrs = cuda_hdrs,
            srcs = native.glob([cuda_so_path(so, version)]),
            strip_include_prefix = cuda_include_prefix,
            visibility = ["//visibility:public"],
            linkopts = [
                "-Wl,--no-as-needed," +
                "-l:lib" + so + ".so," +
                "--as-needed",
            ],
        )

    # NPP
    npp_hdrs = native.glob([cuda_include_prefix + "/npp*.*"])  # FIXME separate out headers

    if family == "aarch64-qnx":
        # Add CULIBOS which is a dependency for NPP libs
        native.cc_library(
            name = "culibos",
            hdrs = cuda_hdrs,
            srcs = ["usr/local/cuda-" + version + "/lib64/libculibos.a"],
            strip_include_prefix = cuda_include_prefix,
            linkstatic = True,
            visibility = ["//visibility:public"],
        )

        native.cc_library(
            name = "nppc",
            hdrs = npp_hdrs,
            srcs = native.glob([cuda_so_path("nppc", version)]),
            # Dependency graph: nppc <- npps <- everything else
            deps = ["cudart"] + ["culibos"],
            strip_include_prefix = cuda_include_prefix,
            linkopts = [
                "-Wl,--no-as-needed," +
                "-l:libnppc.so," +
                "--as-needed",
            ],
            visibility = ["//visibility:public"],
        )

        native.cc_library(
            name = "npps",
            hdrs = npp_hdrs,
            srcs = native.glob([cuda_so_path("npps", version)]),
            # Dependency graph: nppc <- npps <- everything else
            deps = ["cudart"] + ["culibos"] + ["nppc"],
            strip_include_prefix = cuda_include_prefix,
            linkopts = [
                "-Wl,--no-as-needed," +
                "-l:libnpps.so," +
                "--as-needed",
            ],
            visibility = ["//visibility:public"],
        )

        # The lib in NPP_STATIC need to do static link for QNX
        for lib in NPP_STATIC:
            native.cc_library(
                name = lib,
                hdrs = npp_hdrs,
                srcs = [cuda_static_lib_path(lib, version)],
                # Dependency graph: nppc <- npps <- everything else
                deps = ["cudart"] + ["culibos"] + ["nppc"] + ["npps"],
                strip_include_prefix = cuda_include_prefix,
                linkstatic = True,
                visibility = ["//visibility:public"],
            )
    else:
        for so in NPP_SO:
            native.cc_library(
                name = so,
                hdrs = npp_hdrs,
                srcs = native.glob([cuda_so_path(so, version)]),
                # Dependency graph: nppc <- npps <- everything else
                deps = ["cudart"] +
                       ["nppc"] if so != "nppc" else [] +
                                                 ["npps"] if so != "npps" and so != "nppc" else [],
                strip_include_prefix = cuda_include_prefix,
                linkopts = [
                    "-Wl,--no-as-needed," +
                    "-l:lib" + so + ".so," +
                    "--as-needed",
                ],
                visibility = ["//visibility:public"],
            )

    # THRUST
    native.cc_library(
        name = "thrust",
        hdrs = native.glob([cuda_include_prefix + "/thrust/**/*"]),
        deps = ["cudart"],
        strip_include_prefix = cuda_include_prefix,
        visibility = ["//visibility:public"],
    )

    # CUDNN and CUBLAS
    if family == 'aarch64-qnx':
        native.cc_library(
            name = "cudnn",
            hdrs = native.glob(["usr/include/cudnn*.h"]),
            includes = [cuda_include_prefix],
            strip_include_prefix = "usr/include",
            srcs = native.glob(["usr/lib/libcudnn*.so*"]),
            deps = ["cudart"],
            linkstatic = True,
            linkopts = [
            ],
            visibility = ["//visibility:public"],
        )
        native.cc_library(
            name = "cublas",
            hdrs = native.glob(["usr/include/*.h"]),
            srcs = native.glob(["usr/local/cuda-" + version + "/lib64/libcublas*.so*"]),
            strip_include_prefix = "usr/include",
            visibility = ["//visibility:public"],
            linkopts = [
                "-Wl,--no-as-needed," +
                "-l:libcublasLt.so,-l:libcublas.so," +
                "--as-needed",
            ],
        )
    else:
        linkopts = []
        if version == "12.6":
            linkopts.append("-Wl,--no-as-needed,-l:libcudnn.so.9,--as-needed")
        else:
            linkopts.append("-Wl,--no-as-needed,-l:libcudnn.so.8,--as-needed")

        native.cc_library(
            name = "cudnn",
            hdrs = native.glob([cuda_include_prefix + "/cudnn*.h"]),
            includes = [cuda_include_prefix],
            strip_include_prefix = cuda_include_prefix,
            srcs = native.glob([cuda_target_prefix + "/lib64/libcudnn*.so*"]),
            deps = ["cudart"],
            linkstatic = True,
            linkopts = linkopts,
            visibility = ["//visibility:public"],
        )
        native.cc_library(
            name = "cublas",
            hdrs = native.glob([cuda_include_prefix + "/*.h"]),
            srcs = native.glob([cuda_target_prefix + "/lib64/libcublas*.so*"]),
            strip_include_prefix = cuda_include_prefix,
            visibility = ["//visibility:public"],
            linkopts = [
                "-Wl,--no-as-needed," +
                "-l:libcublasLt.so,-l:libcublas.so," +
                "--as-needed",
            ],
        )


# Selects the correct version of `target` based on the current platform
def _cuda_select(target):
    return select({
        "//engine/build:platform_x86_64_cuda_12_2": ["@cuda_x86_64_12020//:" + target],
        "//engine/build:platform_x86_64_cuda_12_6": ["@cuda_x86_64_12060//:" + target],
        "//engine/build:platform_x86_64_rhel9_cuda_12_2": ["@cuda_x86_64_rhel9_12020//:" + target],
        "//engine/build:platform_hp21ea_sbsa": ["@cuda_aarch64_hp21ea_sbsa//:" + target],
        "//engine/build:platform_hp21ga_sbsa": ["@cuda_aarch64_hp21ga_sbsa//:" + target],
        "//engine/build:platform_jetpack60": ["@cuda_aarch64_jetpack60//:" + target],
        "//engine/build:platform_jetpack61": ["@cuda_aarch64_jetpack61//:" + target],
    })

# Creates all CUDA related dependencies for the current platform
def cuda_deps():
    TARGETS = ["cuda", "cublas"] + CUDA_SO + NPP_SO + ["cudnn", "thrust","nvtx"]
    for target in TARGETS:
        native.cc_library(
            name = target,
            visibility = ["//visibility:public"],
            deps = _cuda_select(target),
        )

    # cuda headers only target, to be used by header only deps
    native.cc_library(
        name = "cuda_headers",
        deps = _cuda_select("cuda_headers"),
        visibility = ["//visibility:public"],
    )

def clean_dep(dep):
    return str(Label(dep))

def cuda_workspace():
    """Loads external dependencies required to build apps with alice"""
    # CUDA 12.2 from http://cuda-repo/release-candidates/kitpicks/cuda-r12-2/12.2.1/020/repos/ubuntu2204/sbsa/
    # CUDNN 8.9.2.29 http://cuda-repo/release-candidates/kitpicks/cudnn-v8-9-cuda-12-1/8.9.2.29/001/repos/ubuntu2204/sbsa/
    # repackaged by cuda_cudnn_package_generation.sh
    nv_gxf_http_archive(
        name = "cuda_aarch64_hp21ea_sbsa",
        build_file = clean_dep("//third_party:cuda_aarch64_hp21ea_sbsa.BUILD"),
        sha256 = "f63993349ff37eceae05675a59411cd2274bf94e0aa715faaf7afcbfa997d2b7",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/cuda/cuda12.2-cudnn8.9.2.29-sbsa-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    # CUDA 12.6 from https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.0-560.28.03-1_arm64.deb
    # CUDNN 9.3.0 from https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-repo-ubuntu2204-9.3.0_1.0-1_arm64.deb
    # repackaged by cuda_cudnn_package_generation.sh
    nv_gxf_http_archive(
        name = "cuda_aarch64_hp21ga_sbsa",
        build_file = clean_dep("//third_party:cuda_aarch64_hp21ga_sbsa.BUILD"),
        sha256 = "48fed6cd72206aa065ae7ee6b38c5f3f9d99fe3e96f5a97888d59feada258884",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/cuda/cuda12.6-cudnn9.3.0.75-sbsa-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    # https://sdkm.nvidia.com/products?product=jetson&sdk=JetPack+6.0+DP&tab=src+dirs
    # CUDA 12.2 from http://cuda-repo/release-candidates/kitpicks/cuda-r12-2-tegra/12.2.12/005/local_installers/cuda-tegra-repo-ubuntu2204-12-2-local_12.2.12-1_arm64.deb
    # CUDNN 8.9.4.25 from http://cuda-repo/release-candidates/kitpicks/cudnn-v8-9-tegra/8.9.4.25/001/local_installers/cudnn-local-tegra-repo-ubuntu2204-8.9.4.25_1.0-1_arm64.deb
    # repackaged by cuda_cudnn_package_generation.sh
    nv_gxf_http_archive(
        name = "cuda_aarch64_jetpack60",
        build_file = clean_dep("//third_party:cuda_aarch64_jetpack60.BUILD"),
        sha256 = "735fd74bf36ce305dab226179fc76897ca6f4e3adc4c3f8a41cf6ef38f138045",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/cuda/cuda12.2-cudnn8.9.4.25-arm64-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    # CUDA 12.6 from https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-tegra-repo-ubuntu2204-12-6-local_12.6.0-1_arm64.deb
    # CUDNN 9.3.0 from https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-tegra-repo-ubuntu2204-9.3.0_1.0-1_arm64.deb
    # repackaged by cuda_cudnn_package_generation.sh
    nv_gxf_http_archive(
        name = "cuda_aarch64_jetpack61",
        build_file = clean_dep("//third_party:cuda_aarch64_jetpack61.BUILD"),
        sha256 = "65988373facdfa6a4c57a222bf0043dfb1dd06e5e4d0e5b0fac9b6a07e7abd5a",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/cuda/cuda12.6-cudnn9.3.0.75-arm64-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    # CUDA 12.2.12 from http://cuda-repo/release-candidates/kitpicks/cuda-r12-2-tegra/12.2.12/005/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.12-535.104.05-1_amd64.deb
    # CUDNN 8.9.4.25 from http://cuda-repo/release-candidates/kitpicks/cudnn-v8-9-cuda-12-2/8.9.4.25/001/local_installers/cudnn-local-repo-ubuntu2204-8.9.4.25_1.0-1_amd64.deb
    # repackaged by cuda_cudnn_package_generation.sh
    nv_gxf_http_archive(
        name = "cuda_x86_64_12020",
        build_file = clean_dep("//third_party:cuda_x86_64_12020.BUILD"),
        sha256 = "ff01d8d27bbe1e71e4d8ae7badcac05e859418631f29455b0243576fb2202a6c",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/cuda/cuda12.2-cudnn8.9.4.25-amd64-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    # CUDA 12.6 from https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.0-560.28.03-1_amd64.deb
    # CUDNN 9.3.0 from https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-repo-ubuntu2204-9.3.0_1.0-1_amd64.deb
    # repackaged by cuda_cudnn_package_generation.sh
    nv_gxf_http_archive(
        name = "cuda_x86_64_12060",
        build_file = clean_dep("//third_party:cuda_x86_64_12060.BUILD"),
        sha256 = "a6c39c67009fa23dc52c324f69635784ac85a73e8ef8e5d4e8cb5b70d0ed0087",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/cuda/cuda12.6-cudnn9.3.0.75-amd64-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    # CUDA 12.2.12 from http://cuda-repo/release-candidates/kitpicks/cuda-r12-2-tegra/12.2.12/005/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.12-535.104.05-1_amd64.deb
    # CUDNN 8.9.4.25 from https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.2/local_installers/12.x/cudnn-local-repo-rhel9-8.9.2.26-1.0-1.x86_64.rpm
    # repackaged by cuda_cudnn_package_generation.sh
    nv_gxf_http_archive(
        name = "cuda_x86_64_rhel9_12020",
        build_file = clean_dep("//third_party:cuda_x86_64_rhel9_12020.BUILD"),
        sha256 = "80f9e93b6ec7eaf7efc42e9bdad375f9aeaefbd95c19ccf475b4915bf81a8f84",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/cuda/cuda12.2-cudnn8.9.4.25-amd64-rhel9-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )
