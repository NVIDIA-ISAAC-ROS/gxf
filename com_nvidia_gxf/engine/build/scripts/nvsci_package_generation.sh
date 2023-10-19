#!/usr/bin/env bash
#####################################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################

#####################################################################################
# This is a helper script for generating NvSci binary package from debian archives
# automatic build packages
# While creating package for arm64, x86 package is also required as the include files are
# present only in the x86 package
# First argument is always the target debian package
#
# 1. Usage for x86_64 debian package:
# $ ./nvsci_package_generation.sh <x86 deb package> <platform: x86_64> <version>
#
# 2. Usage for arm64 debian package:
# $ ./nvsci_package_generation.sh <arm64 deb package> <platform: arm64> <version> <x86_64 deb package>
#
# This will generate a package in the form of nvsci_1.0-arm64-tar-xz.xz
#####################################################################################

function extract_debian_package () {
    local debian_package=${1}

    # Create temporary directory
    local temp_directory=$(mktemp -d -t gxf-nvsci-XXXXXXXXXX)
    echo "Using temporary directory path:" $temp_directory

    echo "Extracting deb package into:" ${1} $temp_directory
    dpkg-deb -x $debian_package $temp_directory/deb_root

    extracted_path=$temp_directory/deb_root
}

function generate_nvsci_package () {

    local nvsci_debian=${1}
    # Get platform
    local platform=${2}
    # Get versions
    local nvsci_version=${3}
    local ETC
    local include_path
    local l4t_core_path

    echo "No of arguments passed - $#"

    if [[ $platform == "arm64" ]]; then
        if [[ $# != 6 ]]; then
            echo "Incorrect no of arguments passed - $#"
            echo "Usage: ./nvsci_package_generation.sh <arm64 nvsci deb package> <platform: arm64> <version> <flavor: linux> <arm64 l4t-core debian package> <x86_64 deb package>"
            exit 1
        else
            local flavor=${4}
            local l4t_core_debian=${5}
            local nvsci_x86_debian=${6}
            echo "flavor: $flavor"
        fi
    fi

    if [[ $platform == "x86_64" && $# != 3 ]]; then
        echo "Incorrect no of arguments passed - $#"
        echo "Usage: ./nvsci_package_generation.sh <x86_64 deb package> <platform: x86_64> <version>"
        exit 1
    fi

    if [[ $nvsci_debian =~ $nvsci_version ]]; then
        if [[ $nvsci_debian =~ $platform ]]; then
            echo "Creating nvsci pacakage for version - $nvsci_version, platform - $platform"
        else
            echo "Platform mismatch: Package name - $nvsci_debian, Platform - $platform"
            exit
        fi
    else
        echo "Version mismatch: Package name - $nvsci_debian, Version - $nvsci_version"
        exit
    fi

    export XZ_DEFAULTS="-T0"
    if [[ $platform == "arm64" && $flavor == "linux" ]]; then
        echo "Platform = $platform && flavor = $flavor"
        # Get the include files from the x86_64 debian package
        extract_debian_package $nvsci_x86_debian
        echo "Extracted x86_64 path:" $extracted_path
        include_path=$extracted_path/usr/include

        # Extract l4t-core debian packages which contains nvsci dependencies
        echo "l4t_core_debian - $l4t_core_debian"
        extract_debian_package $l4t_core_debian
        echo "Extracted l4t_core path:" $extracted_path
        l4t_core_path=$extracted_path/usr/lib/aarch64-linux-gnu/tegra

        # Extract arm64 debian package and copy the include file from x86_64 debian package
        extract_debian_package $nvsci_debian
        echo "Extracted path:" $extracted_path
        echo "Copying include files from: "  $include_path
        cp -rf $include_path $extracted_path/usr/

        echo "l4t_core_path = $l4t_core_path"
        cp -rf $l4t_core_path/{libnvrm_mem.so,libnvrm_gpu.so,libnvos.so,libnvsciipc.so,libnvrm_host1x.so,libnvrm_chip.so,libnvrm_sync.so,libnvsocsys.so} \
            $extracted_path/usr/lib/aarch64-linux-gnu/tegra
        platform+="-$flavor"
        echo "Platform = $platform"
    else
        echo "Platform = $platform && flavor = $flavor"
        # Extract x86_64 debian package
        extract_debian_package $nvsci_debian
        ETC=etc
    fi

    echo "Preparing debian packages in the:" $extracted_path
    cd $extracted_path
    local output_path=nvsci_$nvsci_version-$platform-tar-xz.xz
    echo "Creating the archive:" ${output_path}
    tar -cJvf $output_path -C $extracted_path $ETC usr
    cmd="ls $pwd"
    echo $cmd
    cd -
    mv $extracted_path/$output_path ./
    echo "Calculating the checksum:"
    sha256sum ${output_path}

    echo "Removing the temporary directory."
    rm -rf /tmp/gxf-nvsci*
}

# Generate nvsci packages
set -e
generate_nvsci_package $@
