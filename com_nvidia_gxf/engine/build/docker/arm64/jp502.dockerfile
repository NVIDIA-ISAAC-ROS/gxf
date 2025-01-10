#####################################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################

# Full GXF build environment for development activities.
# Base image used for Jetpack 5.0.2  nvcr.io/nvidian/nvidia-l4t-base:r35.1.0
FROM nvcr.io/nvidia/l4t-base@sha256:a3724135352a896f17fee901649aeb584adf8bce7696ad9ad7f0ff263d98fb6e

RUN apt-key adv --fetch-keys http://l4t-repo.nvidia.com/jetson-ota-internal.key
RUN echo "deb http://l4t-repo.nvidia.com/common r35.1 main" >> /etc/apt/sources.list

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y python3-dev python3-pip zlib1g-dev curl git

# Install CUDA packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cuda-cudart-dev-11-4 \
    cuda-command-line-tools-11-4 \
    cuda-minimal-build-11-4 \
    cuda-libraries-dev-11-4 \
    cuda-nvml-dev-11-4 \
    libnpp-dev-11-4 \
    libcusparse-dev-11-4 \
    libcublas-dev-11-4 \
    && rm -rf /var/lib/apt/lists/*

# Install cuDNN
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libcudnn8 \
    libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ADD scripts/requirements.txt engine/build/scripts/
ADD scripts/registry_requirements.txt engine/build/scripts/

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r engine/build/scripts/requirements.txt
RUN python3 -m pip install -r engine/build/scripts/registry_requirements.txt

# Install cupy 11x for jetpack
RUN python3 -m pip install cupy-cuda11x -f https://pip.cupy.dev/aarch64

COPY docker/docker_entrypoint.sh /
ENV USER=root
ENTRYPOINT ["/docker_entrypoint.sh"]
CMD ["/bin/bash"]