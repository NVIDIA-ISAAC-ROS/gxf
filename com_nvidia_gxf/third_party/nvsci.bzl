"""
 SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

load("//third_party:deps.bzl", "local_archive")

NVSCI_SO = [
    "nvscibuf",
    "nvscisync",
    "nvscievent",
]

NVSCI_X86_COMMON_SO = [
    "nvscicommon",
    "nvsciipc",
    "nvsci_mm",
]

NVSCI_NEEDED_SO = [
    "nvos",
    "nvrm_gpu",
    "nvrm_host1x",
    "nvrm_mem",
    "nvrm_sync",
    "nvscicommon",
    "nvsciipc",
]

# Get the path for the shared library with given name for the given version
def nvsci_so_path(name, family):
    if (family == "aarch64"):
        return "usr/lib/" + family + "-linux-gnu/tegra/lib" + name + ".so*"
    else:
        return "usr/lib/" + family + "-linux-gnu" + "/lib" + name + ".so*"

# Creates libraries based on the dependencies specified for the given family
def make_cc_library(so, dependencies, headers, family):
    native.cc_library(
        name = so,
        hdrs = headers,
        srcs = native.glob([nvsci_so_path(so, family)]),
        deps = dependencies,
        strip_include_prefix = "usr/include",
        visibility = ["//visibility:public"],
        linkopts = [
            "-Wl,--no-as-needed," +
            "-l:lib" + so + ".so," +
            "--as-needed",
        ],
    )

# Creates nvsci related dependencies.
def nvsci_device_deps(family):
    nvsci_hdrs = native.glob(["usr/include/*.h"])
    if (family == "aarch64"):
        for so in NVSCI_NEEDED_SO:
            native.cc_library(
                name = so,
                srcs = native.glob([nvsci_so_path(so, family)]),
                visibility = ["//visibility:public"],
                linkopts = [
                    "-Wl,--no-as-needed," +
                    "-l:lib" + so + ".so," +
                    "--as-needed",
                ],
            )

        native.cc_library(
            name = "nvrm_chip",
            srcs = native.glob([nvsci_so_path("nvrm_chip", family)]),
            linkopts = [
                "-Wl,--no-as-needed," +
                "-l:libnvrm_chip.so," +
                "--as-needed",
            ],
            visibility = ["//visibility:public"],
        )

        native.cc_library(
            name = "nvsocsys",
            srcs = native.glob([nvsci_so_path("nvsocsys", family)]),
            linkopts = [
                "-Wl,--no-as-needed," +
                "-l:libnvsocsys.so," +
                "--as-needed",
            ],
            visibility = ["//visibility:public"],
        )

        native.cc_library(
            name = "nvscibuf",
            hdrs = nvsci_hdrs,
            srcs = native.glob([nvsci_so_path("nvscibuf", family)]),
            deps = ["nvrm_mem"] + ["nvsocsys"] + ["nvrm_sync"] + ["nvrm_gpu"] + ["nvos"] + ["nvscicommon"] + ["nvsciipc"],
            strip_include_prefix = "usr/include",
            linkopts = [
                "-Wl,--no-as-needed," +
                "-l:libnvscibuf.so," +
                "--as-needed",
            ],
            visibility = ["//visibility:public"],
        )

        native.cc_library(
            name = "nvscisync",
            hdrs = nvsci_hdrs,
            srcs = native.glob([nvsci_so_path("nvscisync", family)]),
            deps = ["nvrm_host1x"] + ["nvrm_chip"] + ["nvos"] + ["nvscicommon"] + ["nvscibuf"] + ["nvsciipc"],
            strip_include_prefix = "usr/include",
            linkopts = [
                "-Wl,--no-as-needed," +
                "-l:libnvscisync.so," +
                "--as-needed",
            ],
            visibility = ["//visibility:public"],
        )

        native.cc_library(
            name = "nvscievent",
            hdrs = nvsci_hdrs,
            srcs = native.glob([nvsci_so_path("nvscievent", family)]),
            deps = ["nvsciipc"],
            strip_include_prefix = "usr/include",
            linkopts = [
                "-Wl,--no-as-needed," +
                "-l:libnvscievent.so," +
                "--as-needed",
            ],
            visibility = ["//visibility:public"],
        )
    else:
        for so in NVSCI_X86_COMMON_SO:
            native.cc_library(
                name = so,
                srcs = native.glob([nvsci_so_path(so, family)]),
                visibility = ["//visibility:public"],
                linkopts = [
                    "-Wl,--no-as-needed," +
                    "-l:lib" + so + ".so," +
                    "--as-needed",
                ],
            )

        so = "nvscibuf"
        dependencies = ["nvsciipc"] + ["nvscicommon"] + ["nvsci_mm"]
        make_cc_library(so, dependencies, nvsci_hdrs, family)

        so = "nvscisync"
        dependencies = ["nvsciipc"] + ["nvscicommon"] + ["nvscibuf"]
        make_cc_library(so, dependencies, nvsci_hdrs, family)

        so = "nvscievent"
        dependencies = ["nvsciipc"]
        make_cc_library(so, dependencies, nvsci_hdrs, family)

# Selects the correct version of `target` based on the current platform
def _nvsci_select(target):
    return select({
        "//conditions:default": ["@nvsci_x86_64//:" + target],
        "//engine/build:platform_x86_64_cuda_11_8": ["@nvsci_x86_64_cuda_11_8//:" + target],
        "//engine/build:platform_x86_64_cuda_12_1": ["@nvsci_x86_64_cuda_12_1//:" + target],
        "//engine/build:platform_hp11_sbsa": ["@nvsci_aarch64_hp11_sbsa//:" + target],
        "//engine/build:platform_hp20_sbsa": ["@nvsci_aarch64_hp20_sbsa//:" + target],
        "//engine/build:platform_hp21ea_sbsa": ["@nvsci_aarch64_hp21ea_sbsa//:" + target],
        "//engine/build:platform_jetpack51": ["@nvsci_aarch64_jetpack51//:" + target],
    })

# Creates all nvsci related dependencies for the current platform
def nvsci_deps():
    for target in NVSCI_SO:
        native.cc_library(
            name = target,
            visibility = ["//visibility:public"],
            deps = _nvsci_select(target),
        )

def clean_dep(dep):
    return str(Label(dep))

def nvsci_workspace():
    """Loads external dependencies required to build apps"""

    # Debian package is obtained from  //netapp39/linuxbuilds/daily/Embedded/BuildBrain/Desktop/x86-lin64//stage-main/1316/nvsci_pkg_x86_64_stage-main_20230124_32411046.deb
    local_archive(
        name = "nvsci_x86_64",
        build_file = clean_dep("//third_party:nvsci_x86_64.BUILD"),
        src = "//third_party:nvsci_1.0_stage-main_20230124_32411046-x86_64.tar.xz",
    )

    local_archive(
        name = "nvsci_x86_64_cuda_11_7",
        build_file = clean_dep("//third_party:nvsci_x86_64_cuda_11_7.BUILD"),
        src = "//third_party:nvsci_1.0_stage-main_20230124_32411046-x86_64.tar.xz",
    )

    local_archive(
        name = "nvsci_x86_64_cuda_11_8",
        build_file = clean_dep("//third_party:nvsci_x86_64_cuda_11_8.BUILD"),
        src = "//third_party:nvsci_1.0_stage-main_20230124_32411046-x86_64.tar.xz",
    )

    local_archive(
        name = "nvsci_x86_64_cuda_12_1",
        build_file = clean_dep("//third_party:nvsci_x86_64_cuda_12_1.BUILD"),
        src = "//third_party:nvsci_1.0_stage-main_20230124_32411046-x86_64.tar.xz",
    )

    # Debian package is obtained from https://urm.nvidia.com/artifactory/sw-l4t-generic-local/nightly/rel-35/2023-03-27_0046/t186ref/customer_release.tbz2
    # (customer_release.tbz2 -> t186ref_release_aarch64/Jetson_Linux_R35.1.0_aarch64.tbz2 -> Linux_for_Tegra/nv_tegra/l4t_deb_packages/nvidia-l4t-nvsci*.deb)
    local_archive(
        name = "nvsci_aarch64_jetpack50",
        build_file = clean_dep("//third_party:nvsci_aarch64_jetpack50.BUILD"),
        src = "//third_party:nvsci_2_4-arm64.tar.xz",
    )

    local_archive(
        name = "nvsci_aarch64_jetpack502",
        build_file = clean_dep("//third_party:nvsci_aarch64_jetpack502.BUILD"),
        src = "//third_party:nvsci_2_4-arm64.tar.xz",
    )

    # Debian package is obtained from https://urm.nvidia.com/artifactory/sw-l4t-generic-local/nightly/rel-35/2023-03-27_0046/t186ref/customer_release.tbz2
    # (customer_release.tbz2 -> t186ref_release_aarch64/Jetson_Linux_R35.1.0_aarch64.tbz2 -> Linux_for_Tegra/nv_tegra/l4t_deb_packages/nvidia-l4t-nvsci*.deb)
    local_archive(
        name = "nvsci_aarch64_hp11_sbsa",
        build_file = clean_dep("//third_party:nvsci_aarch64_hp11_sbsa.BUILD"),
        src = "//third_party:nvsci_2_4-arm64.tar.xz",
    )

    # Debian package is obtained from https://urm.nvidia.com/artifactory/sw-l4t-generic-local/nightly/rel-35/2023-03-27_0046/t186ref/customer_release.tbz2
    # (customer_release.tbz2 -> t186ref_release_aarch64/Jetson_Linux_R35.1.0_aarch64.tbz2 -> Linux_for_Tegra/nv_tegra/l4t_deb_packages/nvidia-l4t-nvsci*.deb)
    local_archive(
        name = "nvsci_aarch64_hp20_sbsa",
        build_file = clean_dep("//third_party:nvsci_aarch64_hp20_sbsa.BUILD"),
        #sha256 = "4880ce772ab4a95ff7faa343e603872f282e8298f65169abd22bbed3dea24ac4",
        src = "//third_party:nvsci_2_4-arm64.tar.xz",
    )

    # Debian package is obtained from https://urm.nvidia.com/artifactory/sw-l4t-generic-local/nightly/rel-35/2023-03-27_0046/t186ref/customer_release.tbz2
    # (customer_release.tbz2 -> t186ref_release_aarch64/Jetson_Linux_R35.1.0_aarch64.tbz2 -> Linux_for_Tegra/nv_tegra/l4t_deb_packages/nvidia-l4t-nvsci*.deb)
    local_archive(
        name = "nvsci_aarch64_hp21ea_sbsa",
        build_file = clean_dep("//third_party:nvsci_aarch64_hp21ea_sbsa.BUILD"),
        src = "//third_party:nvsci_2_4-arm64.tar.xz",
    )

    local_archive(
        name = "nvsci_aarch64_jetpack51",
        build_file = clean_dep("//third_party:nvsci_aarch64_jetpack51.BUILD"),
        src = "//third_party:nvsci_2_4-arm64.tar.xz",
    )
