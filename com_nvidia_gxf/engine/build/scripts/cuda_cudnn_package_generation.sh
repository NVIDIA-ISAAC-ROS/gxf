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
#   libcudnn8-dev_8.0.5.39-1+cuda11.1_amd64.deb
#   libcudnn8_8.0.5.39-1+cuda11.1_amd64.deb
#
# $ ./cuda_cudnn_package_generation.sh ./ 11.1 8.0.5.39 amd64
#

# This will generate a tar ball, that can be used in the engine.bzl "cuda_x86_64" archive.
#####################################################################################
function generate_cuda_package () {

    local input_directory=${1}
    local cuda_version=${2}
    local cudnn_version=${3}
    local platform=${4}

    str_arr=(${cuda_version//./ })
    local cuda_major=${str_arr[0]}
    local cuda_minor=${str_arr[1]}
    local cuda_version_dash=${cuda_version[@]/\./-}
    local cuda_debian=$(ls $input_directory/cuda-repo*$cuda_version*.deb)
    local cudnn_debian=$(ls $input_directory/libcudnn*$cudnn_version*$cuda_version*.deb)
    echo "Using input cuda debian path:" $cuda_debian
    echo "Using input cudnn debian:" $cudnn_debian
    echo "Using output path:" $output_path


    # Set the output path.
    local output_path=cuda$cuda_version-cudnn$cudnn_version-$platform-tar-xz
    echo "Using output path:" $output_path


    # A list of CUDA packages and dependecies to install.
    # Note, while dpkg doesn't install dependencies automatically,
    # it will issue an error, if a dependency is missing.
    local packages=(
        cuda-minimal-build-$cuda_version_dash
        cuda-cudart-$cuda_version_dash
        cuda-cudart-dev-$cuda_version_dash
        cuda-compiler-$cuda_version_dash
        cuda-nvprune-$cuda_version_dash
        cuda-nvcc-$cuda_version_dash
        cuda-cuobjdump-$cuda_version_dash
        cuda-driver-dev-$cuda_version_dash
        cuda-nvtx-$cuda_version_dash
        cuda-nvrtc-$cuda_version_dash
        cuda-nvrtc-dev-$cuda_version_dash
        cuda-cuxxfilt-$cuda_version_dash
        cuda-cccl-$cuda_version_dash
        cuda-toolkit-$cuda_version_dash-config-common
        cuda-toolkit-$cuda_major-config-common
        cuda-toolkit-config-common
        lib*-$cuda_version_dash
        libcudnn?_$cudnn_version*
    )

    if [ $platform != "arm64" ]; then
        packages+=(cuda-compat-$cuda_version_dash)
    fi

    # A CuDNN DEV package to install. As of libcudnn8-dev it produces an
    # error in the post-installation script that can only be ignored
    # by unpacking the archive, instead of installing it.
    local package_cudnn_dev=(
        libcudnn?-dev_$cudnn_version*
        )

    # A comma-separated list of ignored dependencies
    local ignore_dependencies=build-essential #,openjdk-7-jre,default-jre,libcairo2

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
    # fakeroot dpkg --log=/dev/null --admindir=output/db --instdir=output \
    #     --ignore-depends=$ignore_dependencies -i ${packages[@]/%/*}

    # Extract all the cuda packages to the output folder
    # With 11.3 version the cuda-toolkit-config-common packages have some
    # post installation failures, hence extract the package instead of installation
    for pkg in ${packages[@]/%/*}
    do
        echo "extracting " $pkg
        fakeroot dpkg -x $pkg output
    done

    # Unpack CuDNN DEV package into the output folder.  Note, libcudnn8-dev throws a non-critical
    # post-installation script error, which can't be ignored. Should this error be removed,
    # installation procedure could be merged into generic CUDA and CuDNN installation above.
    fakeroot dpkg -x ${package_cudnn_dev[@]/%/*} output

    # Move CuDNN package into the usr/local/cuda-$cuda_version
    if [ "$platform" = "arm64" ]; then
        mv output/usr/include/aarch64-linux-gnu/* output/usr/local/cuda-$cuda_version/include
        mv output/usr/lib/aarch64-linux-gnu/* output/usr/local/cuda-$cuda_version/lib64
    else
        mv output/usr/include/x86_64-linux-gnu/* output/usr/local/cuda-$cuda_version/include
        mv output/usr/lib/x86_64-linux-gnu/* output/usr/local/cuda-$cuda_version/lib64
    fi
    cd -

    echo "Creating the archive:" ${output_path}
    export XZ_DEFAULTS="-T0"
    tar -cJvf $output_path -C $temp_directory/output usr/local/cuda-$cuda_version

    echo "Calculating the checksum:"
    sha256sum ${output_path}

    echo "Removing the temporary directory."
    rm -rf $temp_directory
}

# Generate CUDA/CuDNN packages
set -e
generate_cuda_package $1 $2 $3 $4
