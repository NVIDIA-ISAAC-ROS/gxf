"""
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

''' Platform configs and versions for various compute stacks
'''

load("@bazel_skylib//lib:selects.bzl", "selects")

def get_platform_arch():
    arch = selects.with_or({
      ("//conditions:default",
       "@com_nvidia_gxf//engine/build:cpu_host"): "x86_64",
      ("@com_nvidia_gxf//engine/build:cpu_aarch64"): "aarch64",
      ("@com_nvidia_gxf//engine/build:cpu_aarch64_sbsa"): "aarch64_sbsa",
    })

    return arch

def get_platform_os():
    os = selects.with_or({
      "//conditions:default" : "linux",
    })

    return os

def get_platform_os_distribution():
    distribution = selects.with_or({
       "//conditions:default": "ubuntu_22.04",
       "@com_nvidia_gxf//engine/build:host_ubuntu_22_04": "ubuntu_22.04",
       "@com_nvidia_gxf//engine/build:host_rhel9": "rhel9",
    })

    return distribution

def get_cuda_version():
    cuda = select({
      "//conditions:default": "12.6",
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_2": "12.2",
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_6": "12.6",
      "@com_nvidia_gxf//engine/build:platform_x86_64_rhel9_cuda_12_2": "12.2",
      "@com_nvidia_gxf//engine/build:platform_hp21ea_sbsa": "12.2",
      "@com_nvidia_gxf//engine/build:platform_hp21ga_sbsa": "12.6",
      "@com_nvidia_gxf//engine/build:platform_jetpack60": "12.2",
      "@com_nvidia_gxf//engine/build:platform_jetpack61": "12.6",
    })

    return cuda

def get_cudnn_version():
    cudnn = select({
      "//conditions:default": "9.3.0",
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_2": "8.9.2",
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_6": "9.3.0",
      "@com_nvidia_gxf//engine/build:platform_x86_64_rhel9_cuda_12_2": "8.9.2",
      "@com_nvidia_gxf//engine/build:platform_hp21ea_sbsa": "8.9.2",
      "@com_nvidia_gxf//engine/build:platform_hp21ga_sbsa": "9.3.0",
      "@com_nvidia_gxf//engine/build:platform_jetpack60": "8.9.2",
      "@com_nvidia_gxf//engine/build:platform_jetpack61": "9.3.0",
    })

    return cudnn

def get_tensorrt_version():
    tensorrt = select({
      "//conditions:default": None,
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_2": "8.6.1",
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_6": "10.3.0",
      "@com_nvidia_gxf//engine/build:platform_x86_64_rhel9_cuda_12_2": "8.6.1",
      "@com_nvidia_gxf//engine/build:platform_hp21ea_sbsa": "8.6.2",
      "@com_nvidia_gxf//engine/build:platform_hp21ga_sbsa": "10.3.0",
      "@com_nvidia_gxf//engine/build:platform_jetpack60": "8.6.2",
      "@com_nvidia_gxf//engine/build:platform_jetpack61": "10.3.0",
    })

    return tensorrt

def get_deepstream_version():
    deepstream = select({
      "//conditions:default": None,
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_2": "7.1",
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_6": "7.1",
      "@com_nvidia_gxf//engine/build:platform_x86_64_rhel9_cuda_12_2": "7.1",
      "@com_nvidia_gxf//engine/build:platform_hp21ea_sbsa": "7.1",
      "@com_nvidia_gxf//engine/build:platform_hp21ga_sbsa": "7.1",
      "@com_nvidia_gxf//engine/build:platform_jetpack60": "7.1",
      "@com_nvidia_gxf//engine/build:platform_jetpack61": "7.1",
      })

    return deepstream

def get_triton_version():
    triton = select({
      "//conditions:default": None,
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_2": "2.39.0",
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_6": "2.49.0",
      "@com_nvidia_gxf//engine/build:platform_x86_64_rhel9_cuda_12_2": "2.39.0",
      "@com_nvidia_gxf//engine/build:platform_hp21ea_sbsa": "2.27.0",
      "@com_nvidia_gxf//engine/build:platform_hp21ga_sbsa": "2.49.0",
      "@com_nvidia_gxf//engine/build:platform_jetpack60": "2.40.0",
      "@com_nvidia_gxf//engine/build:platform_jetpack61": "2.49.0",
      })

    return triton

def get_vpi_version():
    vpi = select({
      "//conditions:default": None,
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_2": "3.0.10",
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_6": "3.0.10",
      "@com_nvidia_gxf//engine/build:platform_x86_64_rhel9_cuda_12_2": "3.0.10",
      "@com_nvidia_gxf//engine/build:platform_hp21ea_sbsa": "2.2.4",
      "@com_nvidia_gxf//engine/build:platform_hp21ga_sbsa": "3.0.10",
      "@com_nvidia_gxf//engine/build:platform_jetpack60": "3.0.10",
      "@com_nvidia_gxf//engine/build:platform_jetpack61": "3.0.10",
      })

    return vpi