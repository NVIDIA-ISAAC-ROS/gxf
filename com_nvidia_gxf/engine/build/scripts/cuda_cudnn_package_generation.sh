#!/bin/bash
#####################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# $ ./cuda_cudnn_package_generation.sh ./ 11.1 8.0.5.39 amd64 deb
#
# For rpm packages ensure that cuda-repo-rhel9-12-2-local-12.2.2_535.104.05-1.x86_64.rpm and
# cudnn-local-repo-rhel9-8.9.4.25-1.0-1.x86_64.rpm are in the current directory
#
# $ ./cuda_cudnn_package_generation.sh ./ 12.2 8.9.4.25 amd64 rpm

# This will generate a tar ball, that can be used in the engine.bzl "cuda_x86_64" archive.
#####################################################################################
function generate_cuda_package () {

    local input_directory=${1}
    local cuda_version=${2}
    local cudnn_version=${3}
    local platform=${4}
    local pkg_type=${5}

    str_arr=(${cuda_version//./ })
    local cuda_major=${str_arr[0]}
    local cuda_minor=${str_arr[1]}
    local cuda_version_dash=${cuda_version[@]/\./-}

    if [[ $platform == "amd64" || $platform == "sbsa" ]]; then
        local cuda_package=$(ls $input_directory/cuda-repo*$cuda_version*.$pkg_type)
    elif [[ $platform == "arm64" ]]; then
        local cuda_package=$(ls $input_directory/cuda-tegra*$cuda_version*.$pkg_type)
    else
        echo -e "Incorrect platform type - $platform"
        exit 1
    fi

    if [[ $input_directory == "./" ]]; then
        input_directory=`pwd`
    fi

    local lib_cudnn_rpm_packages=""
    local package_path=$(realpath $input_directory)
    if [[ $pkg_type == "rpm" ]]; then
        # Extract libcudnn? and libcudnn?-devel packages from cudnn-local-repo-rhel9-*-*.x86_64.rpm
        # Create temp directory for cudnn packages to be extracted
        local temp_cudnn_directory=$(mktemp -d -t gxf-cudnn-XXXXXXXXXX)
        cd $temp_cudnn_directory
        # rhel packages for cudnn does not have CUDA version in file name
        rpm2cpio $package_path/cudnn*$cudnn_version*.$pkg_type | cpio -idmv
        lib_cudnn_rpm_packages=$(find $temp_cudnn_directory -type f -name lib*.rpm | grep -v *samples*)
        cd -
        for rpm in $lib_cudnn_rpm_packages
        do
            echo "Copying cudnn package - " $rpm
            cp $rpm ./
        done
        rm -rf $temp_cudnn_directory
    elif [[ $pkg_type == "deb" ]]; then
        cudnn_package=$(ls $input_directory/libcudnn*$cuda_major*$cudnn_version*.$pkg_type)
    else
        echo -e "Incorrect cudnn package"
        exit 1
    fi

    echo "Using input cuda package path:" $cuda_package
    echo "Using input cudnn package path:" $cudnn_package
    echo "Using output path:" $output_path


    # Set the output path.
    local output_path=cuda$cuda_version-cudnn$cudnn_version-$platform-tar-xz
    echo "Using output path:" $output_path


    # A list of CUDA packages and dependencies to install.
    # Note, while dpkg doesn't install dependencies automatically,
    # it will issue an error, if a dependency is missing.
    # Since CUDA-12.0 onwards, nvcc package is split into three parts
    # nvcc, crt and nvvm
    local packages=(
        cuda-cccl-$cuda_version_dash
        cuda-compiler-$cuda_version_dash
        cuda-crt-$cuda_version_dash
        cuda-cudart-$cuda_version_dash
        cuda-cuobjdump-$cuda_version_dash
        cuda-cuxxfilt-$cuda_version_dash
        cuda-minimal-build-$cuda_version_dash
        cuda-nvcc-$cuda_version_dash
        cuda-nvprune-$cuda_version_dash
        cuda-nvvm-$cuda_version_dash
        cuda-nvtx-$cuda_version_dash
        cuda-nvrtc-$cuda_version_dash
        cuda-toolkit-$cuda_version_dash-config-common
        cuda-toolkit-$cuda_major-config-common
        cuda-toolkit-config-common
    )

    # Naming convention of developer packages are different for debian and rpm
    if [[ $pkg_type == "rpm" ]]; then
        packages+=(
            cuda-cudart-devel-$cuda_version_dash
            cuda-driver-devel-$cuda_version_dash
            cuda-nvrtc-devel-$cuda_version_dash
            libcudnn?-$cudnn_version*
            libcublas-$cuda_version_dash
            libcublas-devel-$cuda_version_dash
            libcufft-$cuda_version_dash
            libcufft-devel-$cuda_version_dash
            libcufile-$cuda_version_dash
            libcufile-devel-$cuda_version_dash
            libcurand-$cuda_version_dash
            libcurand-devel-$cuda_version_dash
            libcusolver-$cuda_version_dash
            libcusolver-devel-$cuda_version_dash
            libcusparse-$cuda_version_dash
            libcusparse-devel-$cuda_version_dash
            libnpp-$cuda_version_dash
            libnpp-devel-$cuda_version_dash
            libnvidia-nscq-535-535.104.05-1.x86_64.rpm
            libnvjitlink-$cuda_version_dash
            libnvjitlink-devel-$cuda_version_dash
            libnvjpeg-$cuda_version_dash
            libnvjpeg-devel-$cuda_version_dash
        )
    else
        packages+=(
            cuda-cudart-dev-$cuda_version_dash
            cuda-driver-dev-$cuda_version_dash
            cuda-nvrtc-dev-$cuda_version_dash
            libcudnn?-cuda-"$cuda_major"_$cudnn_version*
            lib*-$cuda_version_dash
        )
    fi

    if [[ $platform == "amd64" || $platform == "sbsa" ]]; then
        packages+=(cuda-compat-$cuda_version_dash)
    fi

    # A CuDNN DEV package to install. As of libcudnn8-dev it produces an
    # error in the post-installation script that can only be ignored
    # by unpacking the archive, instead of installing it.
    local package_cudnn_dev=""
    if [[ $pkg_type == "rpm" ]]; then
        package_cudnn_dev=(
            libcudnn?-devel-$cudnn_version*
        )
    else
        package_cudnn_dev=(
            libcudnn?-dev-cuda-"$cuda_major"_$cudnn_version*
        )
    fi

    # A comma-separated list of ignored dependencies
    local ignore_dependencies=build-essential #,openjdk-7-jre,default-jre,libcairo2

    # Create temporary directory
    local temp_directory=$(mktemp -d -t gxf-cuda-XXXXXXXXXX)
    echo "Using temporary directory path:" $temp_directory

    echo "Extracting CUDA local repository into:" $temp_directory
    if [[ $pkg_type == "rpm" ]]; then
        cuda_package_real_path=$(realpath $cuda_package)
        mkdir -p $temp_directory/deb_root
        cd $temp_directory/deb_root
        rpm2cpio $cuda_package_real_path | cpio -idmv
        cd -
    else
        dpkg-deb -x $cuda_package $temp_directory/deb_root
    fi

    echo "Preparing $pkg_type packages in directory:" $temp_directory
    for package_file in $input_directory/*.$pkg_type $temp_directory/deb_root/var/*/*.$pkg_type
    do
        ln -r -s $package_file -t $temp_directory
    done

    cd $temp_directory
    # Create output and empty debian packages database
    mkdir -p output/db/{updates,info,triggers}
    touch output/db/{status,diversions,statoverride}

    # Install selected CUDA and CuDNN packages into the output folder.
    # fakeroot dpkg --log=/dev/null --admindir=output/db --instdir=output \
    #     --ignore-depends=$ignore_dependencies -i ${packages[@]/%/*}

    # Extract all the cuda packages to the output folder
    # With 11.3 version the cuda-toolkit-config-common packages have some
    # post installation failures, hence extract the package instead of installation
    if [[ $pkg_type == "rpm" ]]; then
        local package_path=$(pwd)
        cd output
        for rpm in ${packages[@]/%/*}
        do
            echo "extracting " $package_path/$rpm
            rpm2cpio $package_path/$rpm | cpio -idmv
        done
        cd -
    else
        for pkg in ${packages[@]/%/*}
        do
            echo "extracting " $pkg
            fakeroot dpkg -x $pkg output
        done
    fi

    # Unpack CuDNN DEV package into the output folder.  Note, libcudnn8-dev throws a non-critical
    # post-installation script error, which can't be ignored. Should this error be removed,
    # installation procedure could be merged into generic CUDA and CuDNN installation above.
    if [[ $pkg_type == "rpm" ]]; then
        local package_path=$(pwd)
        cd output
        rpm2cpio $package_path/${package_cudnn_dev[@]/%/*} | cpio -idmv
        cd -
    else
        fakeroot dpkg -x ${package_cudnn_dev[@]/%/*} output
    fi

    if [[ $platform == "sbsa" ]]; then
        cd output/usr/local/cuda-$cuda_version
        ln -s targets/sbsa-linux/include include
        ln -s targets/sbsa-linux/lib lib64
        cd -
    fi

    # Move CuDNN package into the usr/local/cuda-$cuda_version
    if [[ $platform == "arm64" || $platform == "sbsa" ]]; then
        mv output/usr/include/aarch64-linux-gnu/* output/usr/local/cuda-$cuda_version/include
        mv output/usr/lib/aarch64-linux-gnu/* output/usr/local/cuda-$cuda_version/lib64
    else
        if [[ $pkg_type == "rpm" ]]; then
            mv output/usr/include/* output/usr/local/cuda-$cuda_version/include
            mv output/usr/lib64/* output/usr/local/cuda-$cuda_version/lib64
        else
            mv output/usr/include/x86_64-linux-gnu/* output/usr/local/cuda-$cuda_version/include
            mv output/usr/lib/x86_64-linux-gnu/* output/usr/local/cuda-$cuda_version/lib64
        fi
    fi
    cd $input_directory

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
generate_cuda_package $1 $2 $3 $4 $5
