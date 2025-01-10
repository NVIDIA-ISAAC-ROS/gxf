"""
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.

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
        repo_name = "dev-4",
        arch = "aarch64",
        distribution = "ubuntu_22.04",
        os = "linux",
        version = "2.6.0",
        cuda = "12.6",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:standard_extension.BUILD"),
    )

    gxf_import_ext(
        name = "std_x86_64_cuda_12_2",
        ext_name = "StandardExtension",
        repo_name = "dev-4",
        arch = "x86_64",
        distribution = "ubuntu_22.04",
        os = "linux",
        version = "2.6.0",
        cuda = "12.2",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:standard_extension.BUILD"),
    )

    gxf_import_ext(
        name = "std_x86_64_cuda_12_6",
        ext_name = "StandardExtension",
        repo_name = "dev-4",
        arch = "x86_64",
        distribution = "ubuntu_22.04",
        os = "linux",
        version = "2.6.0",
        cuda = "12.6",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:standard_extension.BUILD"),
    )

    gxf_import_ext(
        name = "std_x86_64_rhel9_cuda_12_2",
        ext_name = "StandardExtension",
        repo_name = "dev-4",
        arch = "x86_64",
        distribution = "rhel9.3",
        os = "linux",
        version = "2.6.0",
        cuda = "12.2",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:standard_extension.BUILD"),
    )

    gxf_import_ext(
        name = "std_hp21ea_sbsa",
        ext_name = "StandardExtension",
        repo_name = "dev-4",
        arch = "aarch64_sbsa",
        distribution = "ubuntu_22.04",
        os = "linux",
        version = "2.6.0",
        cuda = "12.2",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:standard_extension.BUILD"),
    )

    gxf_import_ext(
        name = "std_hp21ga_sbsa",
        ext_name = "StandardExtension",
        repo_name = "dev-4",
        arch = "aarch64_sbsa",
        distribution = "ubuntu_22.04",
        os = "linux",
        version = "2.6.0",
        cuda = "12.6",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:standard_extension.BUILD"),
    )

    gxf_import_ext(
        name = "test_aarch64",
        ext_name = "TestHelperExtension",
        repo_name = "dev-4",
        arch = "aarch64",
        distribution = "ubuntu_22.04",
        os = "linux",
        version = "2.6.0",
        cuda = "12.6",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:test_extension.BUILD"),
    )

    gxf_import_ext(
        name = "test_x86_64_cuda_12_2",
        ext_name = "TestHelperExtension",
        repo_name = "dev-4",
        arch = "x86_64",
        distribution = "ubuntu_22.04",
        os = "linux",
        version = "2.6.0",
        cuda = "12.2",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:test_extension.BUILD"),
    )

    gxf_import_ext(
        name = "test_x86_64_cuda_12_6",
        ext_name = "TestHelperExtension",
        repo_name = "dev-4",
        arch = "x86_64",
        distribution = "ubuntu_22.04",
        os = "linux",
        version = "2.6.0",
        cuda = "12.6",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:test_extension.BUILD"),
    )

    gxf_import_ext(
        name = "test_hp21ea_sbsa",
        ext_name = "TestHelperExtension",
        repo_name = "dev-4",
        arch = "aarch64_sbsa",
        distribution = "ubuntu_22.04",
        os = "linux",
        version = "2.6.0",
        cuda = "12.2",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:test_extension.BUILD"),
    )

    gxf_import_ext(
        name = "test_hp21ga_sbsa",
        ext_name = "TestHelperExtension",
        repo_name = "dev-4",
        arch = "aarch64_sbsa",
        distribution = "ubuntu_22.04",
        os = "linux",
        version = "2.6.0",
        cuda = "12.6",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:test_extension.BUILD"),
    )

    gxf_import_ext(
        name = "sample_aarch64",
        ext_name = "SampleExtension",
        repo_name = "dev-4",
        arch = "aarch64",
        distribution = "ubuntu_22.04",
        os = "linux",
        version = "1.6.0",
        cuda = "",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:sample_extension.BUILD"),
    )

    gxf_import_ext(
        name = "sample_x86_64",
        ext_name = "SampleExtension",
        repo_name = "dev-4",
        arch = "x86_64",
        distribution = "ubuntu_22.04",
        os = "linux",
        version = "1.6.0",
        cuda = "",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:sample_extension.BUILD"),
    )

    gxf_import_ext(
        name = "sample_sbsa",
        ext_name = "SampleExtension",
        repo_name = "dev-4",
        arch = "aarch64_sbsa",
        distribution = "ubuntu_22.04",
        os = "linux",
        version = "1.6.0",
        cuda = "",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:sample_extension.BUILD"),
    )
