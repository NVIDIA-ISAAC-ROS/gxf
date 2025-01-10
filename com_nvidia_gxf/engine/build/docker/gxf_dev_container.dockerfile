#####################################################################################
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################
# Full GXF build environment for development activities.

# Use cuda-12.6 base image
FROM nvcr.io/nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV GXF_BUILD_PHASE="yes"
ARG gxf_dev

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    gnupg \
    ca-certificates \
    build-essential \
    git \
    libx11-xcb-dev \
    libxkbcommon-dev \
    libwayland-dev \
    libxrandr-dev \
    libegl1-mesa-dev \
    python3 \
    wget \
    python3-distutils \
    python3-apt \
    # Add open GL libraries
    pkg-config \
    libglvnd-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    systemd \
    systemd-sysv \
    dbus \
    dbus-user-session \
    && rm -rf /var/lib/apt/lists/*

# Install vulkan package
RUN wget -O /tmp/vulkan.tar.gz https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/vulkan/vulkan-1.1.123.tar.gz && \
    tar xf /tmp/vulkan.tar.gz && \
    rm -rf /tmp/vulkan.tar.gz

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive  apt-get install -y \
    zlib1g \
    libxml2 \
    tzdata \
    libyaml-cpp-dev \
    gnutls-bin \
    libglew2.2 libssl3 \
    libcurl4 \
    libuuid1 \
    libgles2-mesa \
    gdb bash-completion \
    uuid-dev \
    libglew-dev \
    libssl-dev \
    freeglut3-dev \
    libcurl4-gnutls-dev \
    libxau-dev \
    libxdmcp-dev \
    libxcb1-dev \
    libxext-dev \
    libx11-dev \
    libnss3 \
    linux-libc-dev \
    libnuma1 \
    libnvidia-compute-560 \
    openssl \
    sshfs \
    python3-pip \
    python3-yaml \
    rsyslog \
    libjsoncpp-dev \
    libpython3.10-dev \
    vim  rsync \
    libpython3.10-dev \
    libcairo2 \
    libpango1.0-dev \
    libpangocairo-1.0-0 \
    libx11-6 \
    python3-yaml \
    libc-dev-bin\
    libc6 \
    libcurl3-gnutls \
    libc-bin \
    linux-libc-dev \
    unzip \
    curl && \
    rm -rf /var/lib/apt/lists/* && \
    apt autoremove

# Install cupy-cuda12.6
RUN pip3 install --upgrade pip && \
    pip3 install "https://github.com/cupy/cupy/releases/download/v13.3.0/cupy_cuda12x-13.3.0-cp310-cp310-manylinux2014_x86_64.whl"

# Install UCX package
RUN mkdir -p /tmp/ucx && \
    wget -O /tmp/ucx/ucx_1.17.0_ubuntu_22.04.deb https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/ucx/ucx_1.17.0_ubuntu_22.04.deb --no-check-certificate --no-verbose && \
    wget -O /tmp/ucx/ucx-cuda_1.17.0_ubuntu_22.04.deb https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/ucx/ucx-cuda_1.17.0_ubuntu_22.04.deb --no-check-certificate --no-verbose && \
    cd /tmp/ucx && \
    dpkg -i ucx_1.17.0_ubuntu_22.04.deb && \
    dpkg -i ucx-cuda_1.17.0_ubuntu_22.04.deb && \
    cd .. && \
    rm -rf /tmp/ucx

# Graph Composer
RUN mkdir -p  /opt/graph_composer && \
    wget -O /opt/graph_composer/graph_composer.tar ${gxf_dev} --no-check-certificate --no-verbose && \
    cd /opt/graph_composer && \
    tar -xf graph_composer.tar && \
    ACCEPT_NVIDIA_SOFTWARE_DEVELOPMENT_KITS_EULA=yes dpkg -i graph_composer-dev-4.1.0_x86_64.deb  && \
    cd .. && \
    rm -rf /opt/graph_composer && \
    /opt/nvidia/graph-composer/extension-dev/install_dependencies.sh --allow-root && \
    unset GXF_BUILD_PHASE

COPY nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json

ENV NVIDIA_DRIVER_CAPABILITIES all

ENV USER=root

CMD ["/sbin/init", "systemctl --user enable gxf_server", "systemctl --user start gxf_server"]
