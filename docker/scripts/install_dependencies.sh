#!/bin/bash
#####################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Only for debug purposes
# set -ex

UBUNTU_VERSION=$(lsb_release -rs)
if [[ $UBUNTU_VERSION != "22.04" ]]; then
  echo -e "Incompatible OS "$UBUNTU_VERSION" Please use Ubuntu 22.04"
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

if ! [ -x "$(command -v sudo)" ]; then
  sudo(){
    PYTHON="python3 -m pip install"
    eval "$@"
  }
else
  PYTHON="python3 -m pip install --user"
fi

install_package(){
 echo -e "\e[32m[INFO] Installing packages - $@\e[0m"
 sudo apt-get install -y $@ -qq > /dev/null
 if [ $? -ne 0 ]; then
   echo -e "\e[31m[ERROR] Failed to install packages - $@\e[0m"
   exit 1
 fi
 echo -e "\e[32m[INFO] Successfully installed packages - $@\e[0m"
}


# Updates the list of available packages and their versions to get up-to-date packages
sudo apt-get update
# Install build and deployment tools
install_package lsb-core
# Installs C++ dev tools
install_package git git-lfs git-review build-essential g++ clang-format-14 lcov
# Installs Bazel deps
install_package pkg-config zip zlib1g-dev unzip curl
# Installs ARM building deps
install_package gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
# Installs UUID lib
install_package uuid-runtime
# Installs python3
install_package python3-dev python3-pip python3-yaml python3-venv
# Installs objdump for container builder
install_package binutils
# Install Sphinx deps
install_package latexmk texlive-latex-recommended texlive-latex-extra \
                texlive-fonts-recommended texlive-luatex texlive-xetex
# Install jq for deploy script
install_package jq
# Install NvSci package
if [[ $UBUNTU_VERSION == "22.04" ]]; then
  wget https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic/dependencies/internal/nvsci/nvsci_pkg_x86_64_rel-36_20230807_34016753.deb || \
    { echo "Error:NvSci installer download failed "; exit 1; }
  sudo dpkg -i nvsci_pkg_x86_64_rel-36_20230807_34016753.deb
  rm nvsci_pkg_x86_64_rel-36_20230807_34016753.deb
fi


# Install python packages
$PYTHON --upgrade pip
$PYTHON --upgrade "setuptools==65.7.0" wheel "pyinstaller==6.10.0" "cython<3.0.0"
$PYTHON -r engine/build/scripts/registry_requirements.txt --verbose || exit 1
$PYTHON -r engine/build/scripts/requirements.txt --verbose || exit 1
# Update PATH env variable
[[ ":$PATH:" != *":$HOME/.local/bin:"* ]] && export PATH="${PATH}:$HOME/.local/bin"

# Install cupy-cuda12.6
CUPY_PATH="https://github.com/cupy/cupy/releases/download/v13.3.0/cupy_cuda12x-13.3.0-cp310-cp310-manylinux2014_x86_64.whl"
$PYTHON $CUPY_PATH

# Add symlink to make python3 default
sudo ln -s /usr/bin/python3 /usr/bin/python

# Installs Bazel
TMPFOLDER=$(mktemp -d)
OUTPUT_SH=$TMPFOLDER/bazel.sh
curl -s -L https://github.com/bazelbuild/bazel/releases/download/6.0.0/bazel-6.0.0-installer-linux-x86_64.sh -o $OUTPUT_SH || \
                       { echo "Error:bazel installer download failed "; exit 1; }
chmod +x $OUTPUT_SH || { echo "Error:'$OUTPUT_SH' permission change failed"; exit 1; }
sudo bash $OUTPUT_SH || { echo "Error:'$OUTPUT_SH' execution failed"; exit 1; }

echo "Installation Succeeded"
