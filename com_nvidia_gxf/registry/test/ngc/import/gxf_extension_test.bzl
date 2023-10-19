"""
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//registry/build:gxf_extension.bzl", "gxf_import_ext")


def clean_dep(dep):
    return str(Label(dep))

def import_extensions():
    gxf_import_ext(
        name = "std_aarch64",
        ext_name = "StandardExtension",
        repo_name = "ngc-public-team",
        arch = "aarch64",
        distribution = "ubuntu_20.04",
        os = "linux",
        version = "2.3.0",
        cuda = "11.4",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:standard_extension.BUILD"),
    )

    gxf_import_ext(
        name = "std_x86_64_cuda_12_1",
        ext_name = "StandardExtension",
        repo_name = "ngc-public-team",
        arch = "x86_64",
        distribution = "ubuntu_20.04",
        os = "linux",
        version = "2.3.0",
        cuda = "12.1",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:standard_extension.BUILD"),
    )

    gxf_import_ext(
        name = "std_hp11_sbsa",
        ext_name = "StandardExtension",
        repo_name = "ngc-public-team",
        arch = "aarch64_sbsa",
        distribution = "ubuntu_20.04",
        os = "linux",
        version = "2.3.0",
        cuda = "11.6",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:standard_extension.BUILD"),
    )

    gxf_import_ext(
        name = "std_hp20_sbsa",
        ext_name = "StandardExtension",
        repo_name = "ngc-public-team",
        arch = "aarch64_sbsa",
        distribution = "ubuntu_20.04",
        os = "linux",
        version = "2.3.0",
        cuda = "11.8",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:standard_extension.BUILD"),
    )

    gxf_import_ext(
        name = "test_aarch64",
        ext_name = "TestHelperExtension",
        repo_name = "ngc-public-team",
        arch = "aarch64",
        distribution = "ubuntu_20.04",
        os = "linux",
        version = "2.3.0",
        cuda = "11.4",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:test_extension.BUILD"),
    )

    gxf_import_ext(
        name = "test_x86_64_cuda_12_1",
        ext_name = "TestHelperExtension",
        repo_name = "ngc-public-team",
        arch = "x86_64",
        distribution = "ubuntu_20.04",
        os = "linux",
        version = "2.3.0",
        cuda = "12.1",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:test_extension.BUILD"),
    )

    gxf_import_ext(
        name = "test_hp11_sbsa",
        ext_name = "TestHelperExtension",
        repo_name = "ngc-public-team",
        arch = "aarch64_sbsa",
        distribution = "ubuntu_20.04",
        os = "linux",
        version = "2.3.0",
        cuda = "11.6",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:test_extension.BUILD"),
    )

    gxf_import_ext(
        name = "test_hp20_sbsa",
        ext_name = "TestHelperExtension",
        repo_name = "ngc-public-team",
        arch = "aarch64_sbsa",
        distribution = "ubuntu_20.04",
        os = "linux",
        version = "2.3.0",
        cuda = "11.8",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:test_extension.BUILD"),
    )

    gxf_import_ext(
        name = "sample_aarch64",
        ext_name = "SampleExtension",
        repo_name = "ngc-public-team",
        arch = "aarch64",
        distribution = "ubuntu_20.04",
        os = "linux",
        version = "1.3.0",
        cuda = "",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:sample_extension.BUILD"),
    )

    gxf_import_ext(
        name = "sample_x86_64",
        ext_name = "SampleExtension",
        repo_name = "ngc-public-team",
        arch = "x86_64",
        distribution = "ubuntu_20.04",
        os = "linux",
        version = "1.3.0",
        cuda = "",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:sample_extension.BUILD"),
    )

    gxf_import_ext(
        name = "sample_sbsa",
        ext_name = "SampleExtension",
        repo_name = "ngc-public-team",
        arch = "aarch64_sbsa",
        distribution = "ubuntu_20.04",
        os = "linux",
        version = "1.3.0",
        cuda = "",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:sample_extension.BUILD"),
    )
