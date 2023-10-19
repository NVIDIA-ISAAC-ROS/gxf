#####################################################################################
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################
# Full GXF build environment for development activities.

# this is the latest known good version from the 11.6-devel-ubuntu20.04 tag
FROM nvcr.io/nvidia/cudagl@sha256:ad5f497710f684a9a66e953dbfea4bb2f1701a3af87ad48b96639294bf92e1e0

# disable terminal interaction for apt
ENV DEBIAN_FRONTEND=noninteractive

# Fix repository key
# Reason: 4/27/22 key changes (https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/)
# Unfortunately the instructions above do not work in docker and we need the following workaround:
RUN apt-key del 7fa2af80 && \
    rm /etc/apt/sources.list.d/cuda.list && \
    apt-get update && apt-get install -y --no-install-recommends wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    rm cuda-keyring_1.0-1_all.deb && \
    rm -rf /var/lib/apt/lists/*

# weget: for vulkan and cudnn download
# apt-utils: https://github.com/phusion/baseimage-docker/issues/319
# aha, rsync, xz-utils: Install AHA utility to prepare HTML reports from the test results

RUN apt-get update && apt-get install -y --no-install-recommends wget git apt-utils aha \
    rsync xz-utils lsb-release software-properties-common && \
    rm -rf /var/lib/apt/lists/*

# Install cuDNN v8 as described in https://hub.docker.com/r/kaixhin/cudnn/dockerfile
ENV CUDNN_RUN_PKG=libcudnn8_8.2.2.26-1+cuda11.4_amd64.deb
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/$CUDNN_RUN_PKG && \
  dpkg -i $CUDNN_RUN_PKG && rm -rf $CUDNN_RUN_PKG

ENV CUDNN_DEV_PKG=libcudnn8-dev_8.2.2.26-1+cuda11.4_amd64.deb
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/$CUDNN_DEV_PKG && \
  dpkg -i $CUDNN_DEV_PKG && rm -rf $CUDNN_DEV_PKG

# Configure the build for CUDA configuration
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Install SDK dependencies
ADD scripts/install_dependencies.sh engine/build/scripts/
ADD scripts/requirements.txt engine/build/scripts/
ADD scripts/registry_requirements.txt engine/build/scripts/
RUN apt update && engine/build/scripts/install_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y python3.8 python3-venv python3-pip python3.8-venv
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10

# Install cupy from whl file
RUN apt update && \
    pip3 install "https://files.pythonhosted.org/packages/f6/9c/cddd95449ef74e8275b00d07d517d6e353b8c4be6ac2e09dda0b5e19a0fe/cupy_cuda116-10.3.0-cp38-cp38-manylinux1_x86_64.whl"
RUN add-apt-repository -y ppa:deadsnakes/ppa && apt update && apt install -y python3.7 python3-venv python3.7-venv
COPY docker/docker_entrypoint.sh /
ENV USER=root
ENTRYPOINT ["/docker_entrypoint.sh"]
CMD ["/bin/bash"]
