"""
 SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

NVSCI_SO = [
    "nvscibuf",
    "nvscisync",
    "nvscievent",
]

NVSCI_X86_COMMON_SO = [
    "gnat-23",
    "nvscicommon",
    "nvsciipc",
    "nvsci_mm",
    "nvos",
]

NVSCI_NEEDED_SO = [
    "nvos",
    "nvrm_gpu",
    "nvrm_host1x",
    "nvrm_mem",
    "nvrm_sync",
    "nvscicommon",
    "nvsciipc",
    "nvtegrahv",
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
        deps = dependencies + ["nvos", "gnat-23"],
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

        nvscisync_tegra_dependencies = [
            "nvrm_host1x",
            "nvrm_chip",
            "nvos",
            "nvscicommon",
            "nvscibuf",
            "nvsciipc"
        ]
        native.cc_library(
            name = "nvscisync",
            hdrs = nvsci_hdrs,
            srcs = native.glob([nvsci_so_path("nvscisync", family)]),
            deps = select({
                "@com_nvidia_gxf//engine/build:platform_jetpack60": nvscisync_tegra_dependencies + ["nvtegrahv"],
                "@com_nvidia_gxf//engine/build:platform_hp21ea_sbsa": nvscisync_tegra_dependencies + ["nvtegrahv"],
                "@com_nvidia_gxf//engine/build:platform_jetpack61": nvscisync_tegra_dependencies + ["nvtegrahv"],
                "@com_nvidia_gxf//engine/build:platform_hp21ga_sbsa": nvscisync_tegra_dependencies + ["nvtegrahv"],
                "//conditions:default": nvscisync_tegra_dependencies,
            }),
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
        "//conditions:default": ["@nvsci_x86_64_cuda_12_2//:" + target],
        "//engine/build:platform_x86_64_cuda_12_2": ["@nvsci_x86_64_cuda_12_2//:" + target],
        "//engine/build:platform_x86_64_cuda_12_6": ["@nvsci_x86_64_cuda_12_6//:" + target],
        "//engine/build:platform_x86_64_rhel9_cuda_12_2": ["@nvsci_x86_64_cuda_12_2//:" + target],
        "//engine/build:platform_hp21ea_sbsa": ["@nvsci_aarch64_hp21ea_sbsa//:" + target],
        "//engine/build:platform_hp21ga_sbsa": ["@nvsci_aarch64_hp21ga_sbsa//:" + target],
        "//engine/build:platform_jetpack60": ["@nvsci_aarch64_jetpack60//:" + target],
        "//engine/build:platform_jetpack61": ["@nvsci_aarch64_jetpack61//:" + target],
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

    # Debian package is obtained from  \\netapp39\linuxbuilds\daily\Embedded\BuildBrain\Desktop\x86-lin64\rel-36
    nv_gxf_http_archive(
        name = "nvsci_x86_64_cuda_12_2",
        build_file = clean_dep("//third_party:nvsci_x86_64_cuda_12_2.BUILD"),
        sha256 = "51ce61eb91a8bfffed690f6f2d55e93ae9cadb63b263993a9264ef70fa6689b4",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic/dependencies/internal/nvsci_cuda_12_2_x86_64_rel-36_20230807_34016753-xz.xz",
        type = "tar.xz",
        licenses = [""],
    )

    # Debian package is obtained from \\netapp39\projects1\builds\release\Embedded\BuildBrain\Desktop\x86-lin64\rel-36\0003\nvsci_pkg_x86_64_rel-36_20240822_37344366.deb
    nv_gxf_http_archive(
        name = "nvsci_x86_64_cuda_12_6",
        build_file = clean_dep("//third_party:nvsci_x86_64_cuda_12_6.BUILD"),
        sha256 = "e82614e435a81d5a8d9e3c59c43cfa4375810ea87d4e57dba4cb1754507bed86",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/nvsci/nvsci_36-x86_64-tar-xz",
        type = "tar.xz",
        licenses = [""],
    )

    #//linux4tegra/l4t/Releases/Engineering Releases/ER-2023-07-27_rel-36/generic/customer_release.tbz2 (generic_release_aarch64 ->
    # Jetson_Linux_R36.0.0_aarch64.tbz2 -> Linux_for_Tegra -> nv_tegra -> l4t_deb_packages -> nvidia-l4t-nvsci_36.0.0-20230727103555_arm64.deb)
    nv_gxf_http_archive(
        name = "nvsci_aarch64_hp21ea_sbsa",
        build_file = clean_dep("//third_party:nvsci_aarch64_hp21ea_sbsa.BUILD"),
        sha256 = "87302dd326e20d70c79d283ea69148c84dd0a3658014e10f25d72a9063b6b916",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/nvsci/nvsci_arm64-linux-36.0.0-20230727103555-xz.xz",
        type = "tar.xz",
        licenses = [""],
    )

    nv_gxf_http_archive(
        name = "nvsci_aarch64_hp21ga_sbsa",
        build_file = clean_dep("//third_party:nvsci_aarch64_hp21ga_sbsa.BUILD"),
        sha256 = "87302dd326e20d70c79d283ea69148c84dd0a3658014e10f25d72a9063b6b916",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/nvsci/nvsci_arm64-linux-36.0.0-20230727103555-xz.xz",
        type = "tar.xz",
        licenses = [""],
    )

    #//linux4tegra/l4t/Releases/Engineering Releases/ER-2023-07-27_rel-36/generic/customer_release.tbz2 (generic_release_aarch64 ->
    # Jetson_Linux_R36.0.0_aarch64.tbz2 -> Linux_for_Tegra -> nv_tegra -> l4t_deb_packages -> nvidia-l4t-nvsci_36.0.0-20230727103555_arm64.deb)
    nv_gxf_http_archive(
        name = "nvsci_aarch64_jetpack60",
        build_file = clean_dep("//third_party:nvsci_aarch64_jetpack60.BUILD"),
        sha256 = "87302dd326e20d70c79d283ea69148c84dd0a3658014e10f25d72a9063b6b916",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/nvsci/nvsci_arm64-linux-36.0.0-20230727103555-xz.xz",
        type = "tar.xz",
        licenses = [""],
    )

    #//linux4tegra/l4t/Releases/Engineering Releases/ER-2024-08-16_rel-36/generic/customer_release.tbz2 (generic_release_aarch64 ->
    # Jetson_Linux_R36.4.0_aarch64.tbz2 -> Linux_for_Tegra -> nv_tegra -> l4t_deb_packages -> nvidia-l4t-nvsci_36.4.0-20240816120710_arm64.deb)
    nv_gxf_http_archive(
        name = "nvsci_aarch64_jetpack61",
        build_file = clean_dep("//third_party:nvsci_aarch64_jetpack61.BUILD"),
        sha256 = "3a6fbffcee9b3399a33b2bc98022ec8dc205c3570c789f24a3dc171361e0222c",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/nvsci/nvsci_arm64-linux-36.4.0-20240816120710-xz.xz",
        type = "tar.xz",
        licenses = [""],
    )
