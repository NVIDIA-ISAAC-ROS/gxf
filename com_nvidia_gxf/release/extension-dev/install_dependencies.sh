#!/bin/bash
#####################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

echo -e "\e[32m[INFO] Updating apt package lists ... \e[0m"

wait_on_error() {
   if [[ $? -ne 0 ]]; then
     echo ""
     echo "** Install dependencies failed. Try running $0 manually to resolve errors"
     echo "** Press any key to exit"
     read -n 1
   fi
}

[[ "${WAIT_ON_ERROR}" -eq 1 ]] && trap "wait_on_error" EXIT

${SUDO} apt-get update > /dev/null
if [ $? -ne 0 ]; then
    echo -e "\e[31m[ERROR] Failed to update apt package lists\e[0m"
    exit 1
fi

echo -e "\e[32m[INFO] Installing dependencies ... \e[0m"

function install_apt_package() {
  echo -e "\e[21m[INFO] Installing - $@\e[0m"
  ${SUDO} apt-get install -y $@ > /dev/null 2>&1
  if [ $? -ne 0 ]; then
    echo -e "\e[31m[ERROR] Failed to install packages - $@\e[0m"
    exit 1
  fi
 echo -e "\e[32m[INFO] Successfully installed packages - $@\e[0m"
}

function install_pip_package() {
  echo -e "\e[21m[INFO] Installing - $@\e[0m"

  python3 -m pip install $@ > /dev/null 2>&1
  if [ $? -ne 0 ]; then
    echo -e "\e[31m[ERROR] Failed to install pip packages - $@\e[0m"
    exit 1
  fi
 echo -e "\e[32m[INFO] Successfully installed pip packages - $@\e[0m"
}

install_apt_package curl python3-gi python3-pip clang-format \
    g++-9-aarch64-linux-gnu gcc-9-aarch64-linux-gnu unzip git \
    gir1.2-gstreamer-1.0 libgstreamer1.0-dev libffi-dev bison flex \
    libmount-dev gstreamer1.0-x autopoint gettext bison flex meson ninja-build

install_pip_package PyYAML

TMPFOLDER=$(mktemp -d)
OUTPUT_SH=$TMPFOLDER/bazel.sh
curl -s -L https://github.com/bazelbuild/bazel/releases/download/6.0.0/bazel-6.0.0-installer-linux-x86_64.sh -o $OUTPUT_SH || exit 1
chmod +x $OUTPUT_SH || exit 1
${SUDO} bash $OUTPUT_SH > /dev/null || exit 1

echo -e "\e[32m[INFO] Successfully installed Bazel 6.0.0 \e[0m"

echo -e "\e[32m[INFO] Successfully installed all dependencies \e[0m"
