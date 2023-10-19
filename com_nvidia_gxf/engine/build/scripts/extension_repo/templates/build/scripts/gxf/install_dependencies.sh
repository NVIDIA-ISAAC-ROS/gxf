#!/bin/bash
#####################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
if [[ $UBUNTU_VERSION != "20.04" ]]; then
  echo -e "Incompatible OS "$UBUNTU_VERSION" Please use Ubuntu 20.04"
  exit 1
fi

# This script shall not be run with sudo
if [ "$EUID" -eq 0 ] && [ -x "$(command -v sudo)" ]; then
  echo "Please do not run as root"
  exit
fi

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
install_package git git-lfs git-review build-essential g++ clang-format-10 lcov
# Installs Bazel deps
install_package pkg-config zip zlib1g-dev unzip curl
# Installs ARM building deps
install_package gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
# Installs python3
install_package python3-dev python3-venv python3-yaml
# Installs objdump for container builder
install_package binutils
# Install Sphinx deps
install_package latexmk texlive-latex-recommended texlive-latex-extra \
                texlive-fonts-recommended texlive-luatex texlive-xetex
# Install jq for deploy script
install_package jq

# Install python packages
# pip is no longer supported for python 3.6.9
# Install archived version
wget https://bootstrap.pypa.io/pip/3.6/get-pip.py
python3 get-pip.py

$PYTHON --upgrade setuptools wheel pyinstaller

# Add symlink to make python3 default
sudo ln -s /usr/bin/python3 /usr/bin/python

# Installs Bazel
TMPFOLDER=$(mktemp -d)
OUTPUT_SH=$TMPFOLDER/bazel.sh
curl -s -L https://github.com/bazelbuild/bazel/releases/download/6.0.0/bazel-6.0.0-installer-linux-x86_64.sh -o $OUTPUT_SH || exit 1
chmod +x $OUTPUT_SH || exit 1
sudo bash $OUTPUT_SH || exit 1

echo "Installation Succeeded"
