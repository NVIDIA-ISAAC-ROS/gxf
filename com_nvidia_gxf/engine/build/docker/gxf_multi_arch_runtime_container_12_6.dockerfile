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
ENV ACCEPT_NVIDIA_SOFTWARE_DEVELOPMENT_KITS_EULA="yes"
ENV GXF_BUILD_PHASE="yes"

ARG gxf_runtime
ARG gxf_runtime_tar
ARG TARGETPLATFORM

RUN apt update && \
    apt install -y wget libyaml-cpp-dev && \
    wget ${gxf_runtime} && \
    tar -xvf ${gxf_runtime_tar} && \
    apt install -y python3-pip python3-yaml curl rsync && \
    pip3 install numpy

RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
    apt-get update && \
    apt-get install -y --no-install-recommends apt-utils aha \
    xz-utils lsb-release software-properties-common libnuma-dev && \
    apt-get install -y systemd systemd-sysv dbus dbus-user-session && \
    apt-get install -y sudo dialog && \
    apt-get install debconf-utils libnvidia-compute-560:amd64 && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install "https://github.com/cupy/cupy/releases/download/v13.3.0/cupy_cuda12x-13.3.0-cp310-cp310-manylinux2014_x86_64.whl" && \
    dpkg -i graph_composer-runtime-4.1.0_x86_64.deb && \
    mkdir -p /tmp/ucx && \
    wget -O /tmp/ucx/ucx_1.17.0_ubuntu_22.04.deb https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/ucx/ucx_1.17.0_ubuntu_22.04.deb --no-check-certificate --no-verbose && \
    wget -O /tmp/ucx/ucx-cuda_1.17.0_ubuntu_22.04.deb https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/ucx/ucx-cuda_1.17.0_ubuntu_22.04.deb --no-check-certificate --no-verbose && \
    cd /tmp/ucx && \
    dpkg -i ucx_1.17.0_ubuntu_22.04.deb && \
    dpkg -i ucx-cuda_1.17.0_ubuntu_22.04.deb && \
    cd .. && \
    rm -rf /tmp/ucx && \
    unset GXF_BUILD_PHASE \
    ; fi

RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
    apt update && apt install -y python3 python3-dev zlib1g-dev git libyaml-0-2 libnuma1\
    python3.10 python3-venv python3.10-venv && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install "https://github.com/cupy/cupy/releases/download/v13.3.0/cupy_cuda12x-13.3.0-cp310-cp310-manylinux2014_aarch64.whl" && \
    dpkg -i graph_composer-runtime-4.1.0_arm64.deb && \
    mkdir -p /tmp/ucx && \
    wget -O /tmp/ucx/ucx_1.17.0_jetpack61.deb https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/ucx/ucx_1.17.0_jetpack61.deb --no-check-certificate --no-verbose && \
    wget -O /tmp/ucx/ucx-cuda_1.17.0_jetpack61.deb https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/ucx/ucx-cuda_1.17.0_jetpack61.deb --no-check-certificate --no-verbose && \
    cd /tmp/ucx && \
    dpkg -i ucx_1.17.0_jetpack61.deb && \
    dpkg -i ucx-cuda_1.17.0_jetpack61.deb && \
    cd .. && \
    rm -rf /tmp/ucx && \
    unset GXF_BUILD_PHASE \
    ; fi

RUN rm ${gxf_runtime_tar} graph_composer-runtime-4.1.0_arm64.deb graph_composer-runtime-4.1.0_x86_64.deb

RUN apt update
ENV USER=root
CMD ["/sbin/init", "systemctl --user enable gxf_server", "systemctl --user start gxf_server"]
