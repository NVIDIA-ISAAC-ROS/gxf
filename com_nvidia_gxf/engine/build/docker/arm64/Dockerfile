#####################################################################################
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################
# Full GXF build environment for development activities.

FROM nvcr.io/nvidian/nvidia-aarch64-base:ubuntu22.04

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y python3-dev python3-pip zlib1g-dev curl git

ADD scripts/requirements.txt engine/build/scripts/
ADD scripts/registry_requirements.txt engine/build/scripts/
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r engine/build/scripts/requirements.txt
RUN python3 -m pip install -r engine/build/scripts/registry_requirements.txt
RUN python3 -m pip install pyinstaller==5.13.2

# cupy for arm is not available from pypi yet. Install experimental wheels from cupy github repo.
# See here - https://github.com/cupy/cupy/issues/3196
# RUN python3 -m pip install cupy-cuda102 -f https://github.com/kmaehashi/cupy-jetpack-wheels/releases/download/v10.0.0b3/cupy_cuda102-10.0.0b3-cp37-cp37m-manylinux2014_aarch64.whl

COPY docker/docker_entrypoint.sh /
ENV USER=root
ENTRYPOINT ["/docker_entrypoint.sh"]
CMD ["/bin/bash"]
