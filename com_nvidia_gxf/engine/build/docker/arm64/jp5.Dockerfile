#####################################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################

# Full GXF build environment for development activities.
FROM nvcr.io/nvidian/nvidia-l4t-base:r34.1

RUN apt-key adv --fetch-keys http://l4t-repo.nvidia.com/jetson-ota-internal.key
RUN echo "deb http://l4t-repo.nvidia.com/common r34.1 main" >> /etc/apt/sources.list

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y python3-dev python3-pip zlib1g-dev curl git

RUN apt update && apt-get install -y --no-install-recommends wget && \
    apt-key del 7fa2af80 && \
    wget https://urm.nvidia.com/artifactory/sw-sdkm-jetson-generic-local/5.0_DP/Linux/114/Jetson_50_b114/cuda-repo-l4t-11-4-local_11.4.14-1_arm64.deb && \
    dpkg -i cuda-repo-l4t-11-4-local_11.4.14-1_arm64.deb && \
    rm cuda-repo-l4t-11-4-local_11.4.14-1_arm64.deb && \
    apt-key add /var/cuda-repo-l4t-11-4-local/7fa2af80.pub

RUN apt update && apt-get install -y --no-install-recommends wget && \
    # apt-key del 7fa2af80 && \
    wget https://urm.nvidia.com/artifactory/sw-sdkm-jetson-generic-local/5.0_DP/Linux/114/Jetson_50_b114/cudnn-local-repo-ubuntu2004-8.3.2.49_1.0-1_arm64.deb && \
    dpkg -i cudnn-local-repo-ubuntu2004-8.3.2.49_1.0-1_arm64.deb && \
    rm cudnn-local-repo-ubuntu2004-8.3.2.49_1.0-1_arm64.deb && \
    # apt-key add /var/cuda-repo-l4t-11-4-local/7fa2af80.pub

ADD scripts/requirements.txt engine/build/scripts/
ADD scripts/registry_requirements.txt engine/build/scripts/
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r engine/build/scripts/requirements.txt
RUN python3 -m pip install -r engine/build/scripts/registry_requirements.txt
RUN python3 -m pip install pyinstaller==4.3

# Install cupy 11x for jetpack
RUN python3 -m pip install cupy-cuda11x -f https://pip.cupy.dev/aarch64

COPY docker/docker_entrypoint.sh /
ENV USER=root
ENTRYPOINT ["/docker_entrypoint.sh"]
CMD ["/bin/bash"]
