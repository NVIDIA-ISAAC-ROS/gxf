#!/bin/bash
#####################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#####################################################################################

#####################################################################################
# This is a helper script for generating GXF SDK L4T CUDA/CuDNN binary packages
# from debian archives distributed at the NVIDIA CUDA and CuDNN developer web sites.
#
# Usage:
#  First argument is the directory containing debian file archives for CUDA and CuDNN
#
#
# Sample usage and directory content:
#
# $ ls ./
#   cuda-repo-ubuntu1804-11-1-local_11.1.1-455.32.00-1_amd64.deb
#
# $ ./nvcc_package_generation.sh ./ 11.1 amd64
#

# This will generate a tar ball, that can be used in the engine.bzl "cuda_x86_64" archive.
#####################################################################################
function generate_nvcc_package () {

    local input_directory=${1}
    local cuda_version=${2}
    local platform=${3}

    local cuda_version_dash=${cuda_version[@]/\./-}
    local cuda_debian=$(ls $input_directory/cuda-repo*$cuda_version*.deb)
    echo "Using input cuda debian path:" $cuda_debian
    echo "Using output path:" $output_path


    # Set the output path.
    local output_path=nvcc-$cuda_version-$platform-tar-xz
    echo "Using output path:" $output_path


    # A list of CUDA packages and dependecies to install.
    # Note, while dpkg doesn't install dependencies automatically,
    # it will issue an error, if a dependency is missing.
    local packages=(
        #cuda-driver-dev-$cuda_version_dash
        #cuda-cudart-$cuda_version_dash
        #cuda-cudart-dev-$cuda_version_dash
        cuda-nvcc-$cuda_version_dash
    )

    # A comma-separated list of ignored dependencies
    local ignore_dependencies=build-essential,cuda-driver-dev-$cuda_version_dash,cuda-cudart-$cuda_version_dash,cuda-cudart-dev-$cuda_version_dash

    # Create temporary directory
    local temp_directory=$(mktemp -d -t gxf-cuda-XXXXXXXXXX)
    echo "Using temporary directory path:" $temp_directory

    echo "Extracting CUDA local repository into:" $temp_directory
    dpkg-deb -x $cuda_debian $temp_directory/deb_root

    echo "Preparing debian packages in the:" $temp_directory
    for debian_file in $input_directory/*.deb $temp_directory/deb_root/var/*/*.deb
    do
        ln -r -s $debian_file -t $temp_directory
    done

    cd $temp_directory
    # Create output and empty debian packages database
    mkdir -p output/db/{updates,info,triggers}
    touch output/db/{status,diversions,statoverride}

    # Install sellected CUDA and CuDNN packages into the output folder.
    fakeroot dpkg --log=/dev/null --admindir=output/db --instdir=output \
        --ignore-depends=$ignore_dependencies -i ${packages[@]/%/*}
    cd -

    echo "Creating the archive:" ${output_path} from $temp_directory/output/usr/local/cuda-$cuda_version
    export XZ_DEFAULTS="-T0"
    tar -cJvf $output_path -C $temp_directory/output/usr/local/cuda-$cuda_version bin include lib64 nvvm targets

    echo "Calculating the checksum:"
    sha256sum ${output_path}

    echo "Removing the temporary directory."
    rm -rf $temp_directory
}

# Generate NVCC packages
set -e
generate_nvcc_package $1 $2 $3
