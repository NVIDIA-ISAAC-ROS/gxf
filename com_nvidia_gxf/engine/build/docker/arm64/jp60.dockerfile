#####################################################################################
# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################

# Full GXF build environment for development activities.
# Base image used for Jetpack 6.0
FROM nvcr.io/nvidian/nvidia-l4t-base:r36.0.0

RUN apt-key adv --fetch-keys http://l4t-repo.nvidia.com/jetson-ota-internal.key
RUN echo "deb http://l4t-repo.nvidia.com/common r36.0 main" >> /etc/apt/sources.list
RUN echo 'deb https://repo.download.nvidia.com/jetson/common r36.0.hp main' > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
RUN echo 'deb https://repo.download.nvidia.com/jetson/t234 r36.0.hp main' >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y python3-dev python3-pip zlib1g-dev curl git \
    python3.10 python3-venv python3.10-venv

# Install CUDA packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cuda-cudart-dev-12-2 \
    cuda-command-line-tools-12-2 \
    cuda-minimal-build-12-2 \
    cuda-libraries-dev-12-2 \
    cuda-nvml-dev-12-2 \
    libnpp-dev-12-2 \
    libcusparse-dev-12-2 \
    libcublas-dev-12-2 \
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

# Install cupy 12x for jetpack
RUN python3 -m pip install "cupy-cuda12x<13.0" -f https://pip.cupy.dev/aarch64

COPY docker/docker_entrypoint.sh /
ENV USER=root
ENTRYPOINT ["/docker_entrypoint.sh"]
CMD ["/bin/bash"]
