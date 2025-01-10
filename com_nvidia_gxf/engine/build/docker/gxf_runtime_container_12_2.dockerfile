#####################################################################################
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################
# Full GXF build environment for production containers.

FROM nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04

# disable terminal interaction for apt
ENV DEBIAN_FRONTEND=noninteractive
ENV ACCEPT_NVIDIA_SOFTWARE_DEVELOPMENT_KITS_EULA="yes"
ENV GXF_BUILD_PHASE="yes"

ARG gxf_runtime
ARG gxf_runtime_tar

RUN apt-get update && apt-get install -y --no-install-recommends wget apt-utils aha rsync \
    xz-utils lsb-release software-properties-common libnuma-dev && \
    apt-get install -y sudo && \
    apt-get -y install dialog && \
    apt-get -y install systemd systemd-sysv dbus dbus-user-session && \
    apt-get -y install apt-get install debconf-utils && \
    apt install -y rsync curl python3-pip python3-yaml && \
    pip3 install numpy && \
    rm -rf /var/lib/apt/lists/*

# Install cupy 12.2 from whl file
RUN apt update && \
    pip3 install "https://github.com/cupy/cupy/releases/download/v12.2.0/cupy_cuda12x-12.2.0-cp310-cp310-manylinux2014_x86_64.whl"

RUN apt-get update && \
    wget ${gxf_runtime} && \
    tar -xvf ${gxf_runtime_tar} && \
    apt-get install -y libyaml-cpp-dev && \
    dpkg -i graph_composer-runtime-4.1.0_x86_64.deb && \
    rm ${gxf_runtime_tar} && \
    rm graph_composer-runtime-4.1.0_x86_64.deb

# Install UCX package
RUN mkdir -p /tmp/ucx && \
    wget -O /tmp/ucx/ucx_1.14.1_ubuntu_22.04.deb https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/ucx/ucx_1.14.1_ubuntu_22.04.deb --no-check-certificate --no-verbose && \
    wget -O /tmp/ucx/ucx-cuda_1.14.1_ubuntu_22.04.deb https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/ucx/ucx-cuda_1.14.1_ubuntu_22.04.deb --no-check-certificate --no-verbose && \
    apt-get install -y libnvidia-compute-535 && \
    cd /tmp/ucx && \
    dpkg -i ucx_1.14.1_ubuntu_22.04.deb && \
    dpkg -i ucx-cuda_1.14.1_ubuntu_22.04.deb && \
    cd .. && \
    rm -rf /tmp/ucx  && \
    unset GXF_BUILD_PHASE

RUN apt update
ENV USER=root
CMD ["/sbin/init", "systemctl --user enable gxf_server", "systemctl --user start gxf_server"]
