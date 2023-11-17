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

load("//third_party:deps.bzl", "local_archive")

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
    if family == "aarch64-qnx":
        cuda_target_prefix = "usr/local/cuda-" + version + "/targets/" + family
        cuda_include_prefix = cuda_target_prefix + "/include"
    elif family == 'aarch64_sbsa':
        cuda_target_prefix = "usr/local/cuda-" + version + "/targets/" + "sbsa-linux"
        cuda_include_prefix = cuda_target_prefix + "/include"
    else:
        cuda_target_prefix = "usr/local/cuda-" + version + "/targets/" + family + "-linux"
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
    elif family == 'x86_64':
        native.cc_library(
            name = "cudnn",
            hdrs = native.glob([cuda_include_prefix + "/cudnn*.h"]),
            includes = [cuda_include_prefix],
            strip_include_prefix = cuda_include_prefix,
            srcs = native.glob(["usr/local/cuda-" + version + "/lib64/libcudnn*.so*"]),
            deps = ["cudart"],
            linkstatic = True,
            linkopts = [
                "-Wl,--no-as-needed," +
                "-l:libcudnn.so.8," +
                "--as-needed",
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
    elif family == 'aarch64':
        if version == '11.4':
            native.cc_library(
                name = "cudnn",
                hdrs = native.glob(["usr/include/cudnn*.h"]),
                includes = [cuda_include_prefix],
                strip_include_prefix = "usr/include",
                srcs = native.glob(["usr/local/cuda-11.4/lib64/libcudnn*.so*"]),
                deps = ["cudart"],
                linkstatic = True,
                linkopts = [
                    "-Wl,--no-as-needed," +
                    "-l:libcudnn.so.8," +
                    "--as-needed",
                ],
                visibility = ["//visibility:public"],
            )
            native.cc_library(
                name = "cublas",
                hdrs = native.glob(["usr/local/cuda-11.4/include/*.h"]),
                srcs = native.glob(["usr/local/cuda-11.4/lib64/libcublas*.so*"]),
                strip_include_prefix = "usr/local/cuda-11.4",
                visibility = ["//visibility:public"],
                linkopts = [
                    "-Wl,--no-as-needed," +
                    "-l:libcublasLt.so,-l:libcublas.so," +
                    "--as-needed",
                ],
            )
        else:
            native.cc_library(
                name = "cudnn",
                hdrs = native.glob(["usr/include/cudnn*.h"]),
                includes = [cuda_include_prefix],
                strip_include_prefix = "usr/include",
                srcs = native.glob(["usr/lib/aarch64-linux-gnu/libcudnn*.so*"]),
                deps = ["cudart"],
                linkstatic = True,
                linkopts = [
                    "-Wl,--no-as-needed," +
                    "-l:libcudnn.so.8," +
                    "--as-needed",
                ],
                visibility = ["//visibility:public"],
            )
            native.cc_library(
            name = "cublas",
            hdrs = native.glob(["usr/include/*.h"]),
            srcs = native.glob(["usr/lib/aarch64-linux-gnu/libcublas*.so*"]),
            strip_include_prefix = "usr/include",
            visibility = ["//visibility:public"],
            linkopts = [
                "-Wl,--no-as-needed," +
                "-l:libcublasLt.so,-l:libcublas.so," +
                "--as-needed",
            ],
            )
    elif family == "aarch64_sbsa":
            native.cc_library(
                name = "cudnn",
                hdrs = native.glob([cuda_include_prefix + "/cudnn*.h"]),
                includes = [cuda_include_prefix],
                strip_include_prefix = cuda_include_prefix,
                srcs = native.glob([cuda_target_prefix + "/lib/libcudnn*.so*"]),
                deps = ["cudart"],
                linkstatic = True,
                linkopts = [
                    "-Wl,--no-as-needed," +
                    "-l:libcudnn.so.8," +
                    "--as-needed",
                ],
                visibility = ["//visibility:public"],
            )
            native.cc_library(
            name = "cublas",
            hdrs = native.glob([cuda_include_prefix + "/cublas*.h"]),
            srcs = native.glob([cuda_target_prefix + "/lib/libcublas*.so*"]),
            strip_include_prefix = cuda_include_prefix,
            visibility = ["//visibility:public"],
            linkopts = [
                "-Wl,--no-as-needed," +
                "-l:libcublasLt.so,-l:libcublas.so," +
                "--as-needed",
            ],
            )
    else:
        pass


# Selects the correct version of `target` based on the current platform
def _cuda_select(target):
    return select({
        "//engine/build:platform_x86_64_cuda_11_8": ["@cuda_x86_64_11080//:" + target],
        "//engine/build:platform_x86_64_cuda_12_1": ["@cuda_x86_64_12010//:" + target],
        "//engine/build:platform_hp11_sbsa": ["@cuda_aarch64_hp11_sbsa//:" + target],
        "//engine/build:platform_hp20_sbsa": ["@cuda_aarch64_hp20_sbsa//:" + target],
        "//engine/build:platform_hp21ea_sbsa": ["@cuda_aarch64_hp21ea_sbsa//:" + target],
        "//engine/build:platform_jetpack51": ["@cuda_aarch64_jetpack51//:" + target],
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
    # CUDA 11.4 from cuda-repo-l4t-11-4-local_11.4.14-1_arm64.deb
    # CUDNN 8.3.2.49 from libcudnn8_8.3.1.22-1+cuda11.4_arm64.deb
    # Debian's are obtained from https://urm.nvidia.com/artifactory/sw-sdkm-jetson-generic-local/5.0_DP/Linux/114/Jetson_50_b114/
    # repackaged by cuda_cudnn_package_generation.sh
    nv_gxf_http_archive(
        name = "cuda_aarch64_jetpack50",
        build_file = clean_dep("//third_party:cuda_aarch64_jetpack50.BUILD"),
        sha256 = "a2691e58c2ef47a7183a8f4fd992ba0f54ccc671bf3316f71264bc9802936c4f",
        patches = [clean_dep("//third_party:libcudacxx_aarch64_cuda_11_4.diff")],
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/cuda/cuda11.4-cudnn8.3.2.49-updated-arm64-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    # CUDA 11.4 from cuda-repo-l4t-11-4-local_11.4.14-1_arm64.deb
    # CUDNN 8.4.1.50 from cudnn-local-repo-ubuntu2004-8.4.1.50_1.0-1_arm64.deb
    # Debian's are obtained from https://urm.nvidia.com/artifactory/sw-sdkm-jetson-generic-local/5.0.2/Linux/201/Jetson_502_b201/
    # repackaged by cuda_cudnn_package_generation.sh
    nv_gxf_http_archive(
        name = "cuda_aarch64_jetpack502",
        build_file = clean_dep("//third_party:cuda_aarch64_jetpack502.BUILD"),
        sha256 = "6ef973da37f3d1efb6ac9d36313e86220d2497ab6b07becf8a1926fa65938348",
        patches = [clean_dep("//third_party:libcudacxx_aarch64_cuda_11_4.diff")],
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/cuda/cuda11.4-cudnn8.4.1.50-updated-arm64-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    # CUDA 11.6 from http://cuda-repo.nvidia.com/release-candidates/kitpicks/cuda-r11-6/11.6.1/005/repos/ubuntu2004/sbsa/
    # CUDNN 8.3.3.40 from http://cuda-repo/release-candidates/kitpicks/cudnn-v8-3-cuda-11-5/8.3.3.40/003/repos/ubuntu2004/sbsa/
    # repackaged by cuda_cudnn_package_generation.sh
    nv_gxf_http_archive(
        name = "cuda_aarch64_hp11_sbsa",
        build_file = clean_dep("//third_party:cuda_aarch64_hp11_sbsa.BUILD"),
        sha256 = "e1e3be1ed04dbb0b8f4a2feaf8827c4354ba00c1413e3fdd02edf6793d029ff6",
        patches = [clean_dep("//third_party:libcudacxx_aarch64_cuda_11_6.diff")],
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/cuda/cuda11.6-cudnn8.3.3.40_sbsa-arm64-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    # CUDA 11.8 from http://cuda-internal.nvidia.com/release-candidates/kitpicks/cuda-r11-8/11.8.0/065/local_installers/cuda-repo-cross-sbsa-ubuntu2204-11-8-local_11.8.0-1_all.deb
    # CUDNN 8.6.0 from http://cuda-repo/release-candidates/kitpicks/cudnn-v8-6-cuda-11-8/8.6.0.163/001/repos/ubuntu2004/sbsa/
    # repackaged by cuda_cudnn_package_generation.sh
    nv_gxf_http_archive(
        name = "cuda_aarch64_hp20_sbsa",
        build_file = clean_dep("//third_party:cuda_aarch64_hp20_sbsa.BUILD"),
        sha256 = "60987ce0a75cc97ad210abd92e8f9a9af3c36d235ae161447d76a6f374e7c7ba",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/cuda/cuda11.8-cudnn8.6.0.163-sbsa-arm64-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    # CUDA 12.1 from https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.1-530.30.02-1_arm64.deb
    # CUDNN 8.9.2 from https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.2/local_installers/12.x/cudnn-local-repo-ubuntu2004-8.9.2.26_1.0-1_arm64.deb/
    # repackaged by cuda_cudnn_package_generation.sh
    nv_gxf_http_archive(
        name = "cuda_aarch64_hp21ea_sbsa",
        build_file = clean_dep("//third_party:cuda_aarch64_hp21ea_sbsa.BUILD"),
        sha256 = "3545ff42ecf52815bcf987958522787e1d3e767a892d2df8cb8329ea4ac41b26",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/cuda/cuda12.1-cudnn8.9.2.26-arm64-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    # CUDA 11.4 from cuda-repo-l4t-11-4-local_11.4.19-1_arm64.deb
    # CUDNN 8.6.0 from cudnn-local-tegra-repo-ubuntu2004-8.6.0.166_1.0-1_arm64.deb
    # Debian's are obtained from https://urm.nvidia.com/ui/native/sw-sdkm-jetson-generic-local/5.1/Linux/72/Jetson_51_b72/
    # repackaged by cuda_cudnn_package_generation.sh
    local_archive(
        name = "cuda_aarch64_jetpack51",
        build_file = clean_dep("//third_party:cuda_aarch64_jetpack51.BUILD"),
        patches = [clean_dep("//third_party:libcudacxx_aarch64_cuda_11_4.diff")],
        src = "//third_party:cuda11.4-cudnn8.6.0.166-arm64.tar.xz",
    )

    # CUDA 11.6.2 from https://developer.download.nvidia.com/compute/cuda/11.6.1/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.1-510.47.03-1_amd64.deb
    # CUDNN 8.2.2.26 from http://cuda-repo.nvidia.com/release-candidates/Libraries/cuDNN/v8.2/8.2.2.26_20210701_30141842/11.4.x-r470/Installer/Ubuntu20_04-x64/libcudnn8-dev_8.2.2.26-1+cuda11.4_amd64.deb
    # repackaged by cuda_cudnn_package_generation.sh
    nv_gxf_http_archive(
        name = "cuda_x86_64",
        build_file = clean_dep("//third_party:cuda_x86_64.BUILD"),
        sha256 = "ac1eae69eb2a9f56181fb185e598cb11edaa20c3c3af415b24ee2b6d9c0e7b43",
        patches = [clean_dep("//third_party:libcudacxx_x86_64_cuda_11_6.diff")],
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/cuda/cuda11.6-cudnn8.2.2.26-updated-amd64-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    # CUDA 11.7.1 from http://cuda-repo/release-candidates/kitpicks/cuda-r11-7/11.7.1/017/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.1-515.65.01-1_amd64.deb
    # CUDNN 8.4.1 from https://developer.nvidia.com/compute/cudnn/secure/8.4.1/local_installers/11.6/cudnn-local-repo-ubuntu2004-8.4.1.50_1.0-1_amd64.deb
    # repackaged by cuda_cudnn_package_generation.sh
    # update1 picks up most recent cuda-11.7 package which also includes libcudacxx headers
    nv_gxf_http_archive(
        name = "cuda_x86_64_11071",
        build_file = clean_dep("//third_party:cuda_x86_64_11071.BUILD"),
        sha256 = "f77d81f5b3c32f10e758cd2dc1b371c5cc5cdfc91014b31c868761c0316752fe",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/cuda/cuda11.7-cudnn8.4.1.50-updated-amd64-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    # CUDA 11.8.0 from http://cuda-internal.nvidia.com/release-candidates/kitpicks/cuda-r11-8/11.8.0/065/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
    # CUDNN 8.6.0 from http://cuda-internal.nvidia.com/release-candidates/kitpicks/cudnn-v8-6-cuda-11-8/8.6.0.163/001/repos/ubuntu2004/x86_64/libcudnn8-dev_8.6.0.163-1+cuda11.8_amd64.deb
    # repackaged by cuda_cudnn_package_generation.sh
    local_archive(
        name = "cuda_x86_64_11080",
        build_file = clean_dep("//third_party:cuda_x86_64_11080.BUILD"),
        src = "//third_party:cuda11.8-cudnn8.6.0.163-amd64.tar.xz",
    )

    # CUDA 12.1.0 from https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.1-530.30.02-1_amd64.deb
    # CUDNN 8.8.1 from https://developer.nvidia.com/downloads/compute/cudnn/secure/8.8.1/local_installers/12.0/cudnn-local-repo-ubuntu2004-8.8.1.3_1.0-1_amd64.deb/
    # repackaged by cuda_cudnn_package_generation.sh
    nv_gxf_http_archive(
        name = "cuda_x86_64_12010",
        build_file = clean_dep("//third_party:cuda_x86_64_12010.BUILD"),
        sha256 = "90c479e5f985097b71748c0ef120c9cc151288bf5d36a6dddefb77628f791bea",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/cuda/cuda12.1-cudnn8.8.1.3-amd64-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )
