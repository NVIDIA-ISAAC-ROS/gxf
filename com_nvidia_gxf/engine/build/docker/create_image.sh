#!/bin/bash
#####################################################################################
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################

# Create Docker image tagged "nvcr.io/nvidian/gxf/gxf" for building GXF targets.
docker build -t nvcr.io/nvidian/gxf/gxf -f engine/build/docker/Dockerfile engine/build
