#!/bin/bash
#####################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

function usage() {
    echo "Usage: $0 [--allow-root|-r]"
}

SUDO="sudo"
WAIT_ON_ERROR=0
while [[ $# -gt 0 ]]; do
  arg="$1"
  case "$arg" in
        -r|--allow-root)
        SUDO=""
        shift;;
        --wait-on-error)
        WAIT_ON_ERROR=1
        shift;;
        -h|--help)
        usage
        exit 1;;
        *) echo "ERROR: Unknown argument $arg"; usage; exit 1
  esac
done

if [ "$SUDO" = "sudo" ] && [ "$EUID" -eq 0 ]; then
  echo -e "\e[31m[ERROR] Please do not run as root. If running inside a docker use --allow-root \e[0m"
  exit
fi

echo -e "\e[32m[INFO] Updating dnf package lists ... \e[0m"

wait_on_error() {
   if [[ $? -ne 0 ]]; then
     echo ""
     echo "** Install dependencies failed. Try running $0 manually to resolve errors"
     echo "** Press any key to exit"
     read -n 1
   fi
}

[[ "${WAIT_ON_ERROR}" -eq 1 ]] && trap "wait_on_error" EXIT

${SUDO} yum check-update
if [ $? -eq 1 ]; then
    echo -e "\e[31m[ERROR] Failed to update rpm package lists\e[0m"
    exit 1
fi

echo -e "\e[32m[INFO] Installing dependencies ... \e[0m"

function install_rpm_package() {
  echo -e "\e[21m[INFO] Installing - $@\e[0m"
  ${SUDO} dnf install -y $@ > /dev/null 2>&1
  if [ $? -ne 0 ]; then
    echo -e "\e[31m[ERROR] Failed to install packages - $@\e[0m"
    exit 1
  fi
 echo -e "\e[32m[INFO] Successfully installed packages - $@\e[0m"
}

function build_install_python3_10_package() {
  echo -e "\e[21m[INFO] Pulling python3.10.12 source package and building - $@\e[0m"
  TMPFOLDER=$(mktemp -d)
  cd $TMPFOLDER && wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz \
    && tar xf Python-3.10.12.tgz \
    && cd Python-3.10.12 \
    && ./configure --enable-loadable-sqlite-extensions --enable-shared --enable-optimizations \
    && make -s -j ${nproc} \
    && sudo make altinstall

  if [ $? -ne 0 ]; then
    echo -e "\e[31m[ERROR] Failed to build and install python3.10 - $@\e[0m"
    exit 1
  fi

  sudo unlink /usr/bin/python3
  sudo ln -s /usr/local/bin/python3.10 /usr/bin/python3
  sudo cp -av /usr/local/lib/libpython3.10.so* /usr/lib64/

  cd - && sudo rm -rf $TMPFOLDER
  echo -e "\e[32m[INFO] Successfully installed python3.10 - $@\e[0m"
}

function install_pip_package() {
  echo -e "\e[21m[INFO] Installing - $@\e[0m"

  python3.10 -m pip install $@ > /dev/null 2>&1
  if [ $? -ne 0 ]; then
    echo -e "\e[31m[ERROR] Failed to install pip packages - $@\e[0m"
    exit 1
  fi
 echo -e "\e[32m[INFO] Successfully installed pip packages - $@\e[0m"
}

install_rpm_package curl unzip git libffi-devel sqlite-devel libsqlite3x-devel.x86_64 yaml-cpp-devel

build_install_python3_10_package

install_pip_package pyaml numpy

# Create symlink gcc-11 from gcc
if [ ! -L "/usr/bin/gcc-11" ]; then
    sudo ln -s /usr/bin/gcc /usr/bin/gcc-11;
fi

if [ ! -f "/usr/local/bin/bazel" ]; then
  TMPFOLDER=$(mktemp -d)
  OUTPUT_SH=$TMPFOLDER/bazel.sh
  curl -s -L https://github.com/bazelbuild/bazel/releases/download/6.0.0/bazel-6.0.0-installer-linux-x86_64.sh -o $OUTPUT_SH || exit 1
  chmod +x $OUTPUT_SH || exit 1
  ${SUDO} bash $OUTPUT_SH > /dev/null || exit 1
fi

echo -e "\e[32m[INFO] Successfully installed Bazel 6.0.0 \e[0m"

echo -e "\e[32m[INFO] Successfully installed all dependencies \e[0m"
