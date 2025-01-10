#####################################################################################
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################
# Full GXF build environment for production containers.

FROM nvcr.io/nvidia/cuda:12.6.0-devel-ubuntu22.04

# disable terminal interaction for apt
ENV DEBIAN_FRONTEND=noninteractive

# weget: for vulkan and cudnn download
# apt-utils: https://github.com/phusion/baseimage-docker/issues/319
# aha, rsync, xz-utils: Install AHA utility to prepare HTML reports from the test results
RUN apt-get update && apt-get install -y --no-install-recommends wget apt-utils aha rsync \
    xz-utils lsb-release software-properties-common libnuma-dev && \
    rm -rf /var/lib/apt/lists/*

# Configure the build for CUDA configuration
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Install nsight systems to help profile applications
RUN apt-get update && apt-get install -y \
    cuda-nsight-systems-12-6

# Install SDK dependencies
ADD scripts/install_dependencies.sh engine/build/scripts/
ADD scripts/requirements.txt engine/build/scripts/
ADD scripts/registry_requirements.txt engine/build/scripts/
RUN apt update && engine/build/scripts/install_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

# Install cupy 12.6 from whl file
RUN apt update && \
    pip3 install "https://github.com/cupy/cupy/releases/download/v13.3.0/cupy_cuda12x-13.3.0-cp310-cp310-manylinux2014_x86_64.whl"

# Set cache dir in /tmp/ since we dont have access to {$HOME} in CI
ENV CUPY_CACHE_DIR /tmp/cupy_cuda_12_6/

RUN apt update && apt install -y python3.10 python3-venv python3.10-venv

# Code coverage reporting dependencies
RUN apt-get clean \
    && apt update --fix-missing \
    && apt install -y python3-pip default-jdk && \
    rm -rf /var/lib/apt/lists/*
RUN update-alternatives --config javac
RUN pip3 install lcov_cobertura==2.0.2

COPY docker/docker_entrypoint.sh /
ENV USER=root
ENTRYPOINT ["/docker_entrypoint.sh"]
CMD ["/bin/bash"]
