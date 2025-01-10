#!/usr/bin/env bash
#####################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################

#####################################################################################
# This is a helper script for generating GXF SDK TensorRT binary packages from
# debian archives. First argument is the URL to the server containing debian file
# archives, e.g. :
#  tensorrt_6.0.1.9-1+cuda10.0_arm64.deb, libnvinfer6_6.0.1-1+cuda10.0_arm64.deb, ...
# Example usage: ./engine/build/scripts/tensorrt_package_generation.sh
# * https://sdkm-a.nvidia.com/builds/SDKManager/JetPack_SDKs/4.3/L4T/78_19316_27599411/JETPACK_43_b78/DLA
# * https://sdkm-a.nvidia.com/builds/SDKManager/JetPack_SDKs/4.3/L4T/78_19316_27599411/JETPACK_43_b78/NoDLA
# * http://cuda-repo/release-candidates/Libraries/TensorRT/v6.0/6.0.1.5-cl27267773-eed615fe/10.0-r400/Ubuntu18_04-x64/deb
# This will generate a tar ball that can be uploaded as needed.
# There is currently no support for external sources, the assumption is that this
# script will run on a build system internal to NVIDIA. But the same debian files are
# available on systems external to NVIDIA and could be found at NVIDIA TensorRT
# redistribution sites, please refer to:
#  * https://developer.nvidia.com/tensorrt
#  * https://developer.nvidia.com/embedded/jetpack
#####################################################################################
set -e

# Generates GXF SDK TensorRT binary packages from debian archives. First argument
# is the URL to the server. This server should containing debian file archives, e.g.:
#  tensorrt_6.0.1.9-1+cuda10.0_arm64.deb, libnvinfer6_6.0.1-1+cuda10.0_arm64.deb, ...
function generate_package () {
    local http_link=${1}
    local temp_directory=$(mktemp -d -t trt-XXXXXXXXXX)
    echo "Using temporary directory path:" $temp_directory

    echo "Downloading TensorRT from:" $http_link
    wget -q $http_link -A "libnvinfer*" -A "libnvparsers*" -A "libnvonnxparsers*" -A "tensorrt_*" --show-progress --no-check-certificate -r -l 1 -nd -P $temp_directory

    # Determine the output path. Use $output_path parameter, if available
    local tensorrt_version=$(ls $temp_directory/tensorrt_* | sed -e 's/\.deb$//')
    local platform=$(basename $(dirname $http_link | tr '[:upper:]' '[:lower:]'))
    local variant=$(basename $http_link | tr '[:upper:]' '[:lower:]')
    local output_path=${tensorrt_version}-${variant}-tar-xz
    echo "Using output path:" $output_path

    echo "Extracting TensorRT into:" ${temp_directory}/deb_root
    for debian_file in ${temp_directory}/*.deb
    do
        dpkg-deb -x $debian_file ${temp_directory}/deb_root
    done

    echo "Creating the archive:" ${output_path}
    tar -cJvf ${output_path} -C ${temp_directory}/deb_root usr/include usr/lib usr/src/tensorrt/bin

    echo "Calculating the checksum:"
    sha256sum ${output_path}

    mv ${output_path} ./
    echo "Removing the temporary directory."
    rm -rf ${temp_directory}
}

# Generate TensorRT packages
generate_package $1
