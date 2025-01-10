"""
 SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

load("@io_bazel_rules_docker//container:container.bzl", "container_pull")

# This pulls in Docker base images used for building apps in this repo
def docker_images():
    # x86_64 Dockerimage with cuda 12.2 + gxf dependencies
    # Built using gxf/engine/build/docker/cuda12_2.dockerfile
    # This image is to be used internally to build and run GXF apps on x86 platform
    # tag = nvcr.io/nvidian/gxf-build:4.1.0
    container_pull(
        name = "gxf_docker_image_cuda12_2",
        registry = "urm.nvidia.com",
        repository = "sw-gxf-docker/gxf-build",
        digest = "sha256:60110dad27a718e214aa505b904df8140338372abc68b8fd1d8cf079d8984286",
        timeout = 6000,
    )

    # aarch64 Dockerimage for jetpack60 with cuda 12.2 + gxf dependencies
    # x86_64 Dockerimage with cuda 12.2 + gxf dependencies
    # Built using gxf/engine/build/docker/arm64/jp60.dockerfile
    # This image is to be used internally to build and run GXF apps on l4t platform
    # tag = nvcr.io/nvidian/gxf-l4t-build:4.1.0
    container_pull(
        name = "gxf_docker_image_aarch64_jp60",
        registry = "urm.nvidia.com",
        repository = "sw-gxf-docker/gxf-l4t-build",
        digest = "sha256:81963803c86a3375e2d06960c7d631181b02fb0169f65bdc0e04d4181d5e98bc",
        timeout = 6000,
    )
