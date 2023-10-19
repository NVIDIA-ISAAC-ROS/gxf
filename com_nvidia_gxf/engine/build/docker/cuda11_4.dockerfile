#####################################################################################
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################
# Full Isaac SDK build environment for production containers.

FROM nvcr.io/nvidia/cuda:11.4.1-devel-ubuntu18.04

# disable terminal interaction for apt
ENV DEBIAN_FRONTEND=noninteractive

# Fix repository key
# Reason: 4/27/22 key changes (https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/)
# Unfortunately the instructions above do not work in docker and we need the following workaround:
RUN apt-key del 7fa2af80 && \
    rm /etc/apt/sources.list.d/cuda.list && \
    apt-get update && apt-get install -y --no-install-recommends wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    rm cuda-keyring_1.0-1_all.deb && \
    rm -rf /var/lib/apt/lists/*

# weget: for vulkan and cudnn download
# apt-utils: https://github.com/phusion/baseimage-docker/issues/319
# aha, rsync, xz-utils: Install AHA utility to prepare HTML reports from the test results
RUN apt-get update && apt-get install -y --no-install-recommends wget apt-utils aha rsync xz-utils lsb-release && \
    rm -rf /var/lib/apt/lists/*

# Configure the build for CUDA configuration
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Install SDK dependencies
ADD scripts/install_dependencies.sh engine/build/scripts/
ADD scripts/requirements.txt engine/build/scripts/
ADD scripts/registry_requirements.txt engine/build/scripts/
RUN apt update && engine/build/scripts/install_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

# Install cupy from whl file
RUN apt update && \
    pip3 install "https://files.pythonhosted.org/packages/16/8e/99122b09985f72bf3740608f1c36be29328f63528c6a49c447bd3740c692/cupy_cuda114-9.4.0-cp36-cp36m-manylinux1_x86_64.whl"

COPY docker/docker_entrypoint.sh /
ENV USER=root
ENTRYPOINT ["/docker_entrypoint.sh"]
CMD ["/bin/bash"]
