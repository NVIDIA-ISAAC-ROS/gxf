"""
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

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
       "//conditions:default": "ubuntu_20.04",
    })

    return distribution

def get_cuda_version():
    cuda = select({
      "//conditions:default": "11.8",
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_11_8": "11.8",
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_1": "12.1",
      "@com_nvidia_gxf//engine/build:platform_hp11_sbsa": "11.6",
      "@com_nvidia_gxf//engine/build:platform_hp20_sbsa": "11.8",
      "@com_nvidia_gxf//engine/build:platform_hp21ea_sbsa": "12.1",
      "@com_nvidia_gxf//engine/build:platform_jetpack51": "11.4",
    })

    return cuda

def get_cudnn_version():
    cudnn = select({
      "//conditions:default": "8.2.2.26",
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_11_8": "8.6.0",
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_1": "8.8.1",
      "@com_nvidia_gxf//engine/build:platform_hp11_sbsa": "8.3.3",
      "@com_nvidia_gxf//engine/build:platform_hp20_sbsa": "8.6.0",
      "@com_nvidia_gxf//engine/build:platform_hp21ea_sbsa": "8.9.2",
      "@com_nvidia_gxf//engine/build:platform_jetpack51": "8.6.0",
    })

    return cudnn

def get_tensorrt_version():
    tensorrt = select({
      "//conditions:default": None,
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_11_8": "8.5.1",
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_1": "8.5.3",
      "@com_nvidia_gxf//engine/build:platform_hp11_sbsa": "8.2.3",
      "@com_nvidia_gxf//engine/build:platform_hp20_sbsa": "8.5.1",
      "@com_nvidia_gxf//engine/build:platform_hp21ea_sbsa": "8.6.1",
      "@com_nvidia_gxf//engine/build:platform_jetpack51": "8.5.2",
    })

    return tensorrt

def get_deepstream_version():
    deepstream = select({
      "//conditions:default": None,
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_11_8": "6.2",
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_1": "6.2.1",
      "@com_nvidia_gxf//engine/build:platform_hp11_sbsa": "6.1.1",
      "@com_nvidia_gxf//engine/build:platform_hp20_sbsa": "6.2.1",
      "@com_nvidia_gxf//engine/build:platform_hp21ea_sbsa": "6.2.1",
      "@com_nvidia_gxf//engine/build:platform_jetpack51": "6.2",
      })

    return deepstream

def get_triton_version():
    triton = select({
      "//conditions:default": None,
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_11_8": "2.26.0",
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_1": "2.32.0",
      "@com_nvidia_gxf//engine/build:platform_hp11_sbsa": "2.24.0",
      "@com_nvidia_gxf//engine/build:platform_hp20_sbsa": "2.27.0",
      "@com_nvidia_gxf//engine/build:platform_hp21ea_sbsa": "2.27.0",
      "@com_nvidia_gxf//engine/build:platform_jetpack51": "2.30.0",
      })

    return triton

def get_vpi_version():
    vpi = select({
      "//conditions:default": None,
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_11_8": "2.1.6",
      "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_1": "2.3.1",
      "@com_nvidia_gxf//engine/build:platform_hp11_sbsa": "2.1.5",
      "@com_nvidia_gxf//engine/build:platform_hp20_sbsa": "2.2.4",
      "@com_nvidia_gxf//engine/build:platform_hp21ea_sbsa": "2.2.4",
      "@com_nvidia_gxf//engine/build:platform_jetpack51": "2.1.6",
      })

    return vpi