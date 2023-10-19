#!/bin/bash
#####################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -o errexit -o pipefail -o noclobber -o nounset

UNAME=nvidia

### default arguments

# whether this is a local installation, i.e. it is executed directly on the device, or in a
# cross-platform docker container, default is to ssh to a remote host
LOCAL_INSTALL=false

# for remote install, host name or IP to connect to
HOST=""

# get command line arguments
while [ $# -gt 0 ]
do
  case "$1" in
    -h|--host)
      HOST="$2"
      shift 2
      ;;
    -l|--local)
      LOCAL_INSTALL=true
      shift
      ;;
    -u|--user)
      UNAME="$2"
      shift 2
      ;;
    *)
      echo "Error: Invalid arguments: $1 $2\n"
      exit 1
  esac
done

if [ -z "${HOST}" -a "${LOCAL_INSTALL}" = false ]
then
  echo "Error: Jetson device IP must be specified with -h IP."
  exit 1
fi

# This function will be ran on the target device
remote_function() {
  local IS_LOCAL="$1"

  # Install packages
  sudo apt update && sudo apt install -y rsync curl python3-pip python3-yaml
  python3 -m pip install numpy==1.23.5
}

if [ "${LOCAL_INSTALL}" = true ]
then
  remote_function "${LOCAL_INSTALL}"
else
  # Installs dependencies on Jetson devices
  ssh -t $UNAME@$HOST "$(declare -pf remote_function); remote_function false"

  echo "Rebooting the Jetson device"
fi

