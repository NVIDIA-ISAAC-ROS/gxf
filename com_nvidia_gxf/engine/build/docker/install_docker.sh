#!/bin/bash
#####################################################################################
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################
# Install latest Docker CE on your machine from the Docker repository.
# You may find further information about the Docker installation process online at:
# https://docs.docker.com/install/linux/docker-ce/ubuntu/

# Update the APT sources with the Docker repository. Install Docker. Restart the  Docker service.
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
sudo pkill -HUP dockerd

# Allow your username to use Docker. To run Docker without root privileges.
sudo usermod -aG docker ${USER}

# If you want to update group privileges without logging out and back in, you can run:
#su - ${USER}

# If you want to check Docker's service status, you can run:
#sudo systemctl status docker
