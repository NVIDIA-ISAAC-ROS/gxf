#!/bin/bash
#####################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
set -e

SCRIPT_DIR=$(dirname $(realpath $0))
echo "Script executed from: ${SCRIPT_DIR}"

# Create folder to install app graph
APP_FOLDER=$(mktemp -d -p $SCRIPT_DIR {APP_NAME}.XXXXX)

# Find gxe executable
EXECUTABLE=gxe
EXECUTABLE_PATH="$(find . -name $EXECUTABLE -type d -follow)/$EXECUTABLE"

if [ ! -f $EXECUTABLE_PATH ]; then
  echo "Could not find $EXECUTABLE"
  exit 1
fi

# Find registry cli tool
REGISTRY=registry_cli
REGISTRY_PATH="$(find . -name $REGISTRY -follow)"
if [ ! -f $REGISTRY_PATH ]; then
  echo "Could not find $REGISTRY_PATH"
  exit 1
fi

# Parse command line args
for ARG in "$@"; do
  shift
  if [ $ARG = "--profile" ]; then
    PROFILE=true
    continue
  elif [ $ARG = "--export" ]; then
    EXPORT=true
    continue
  elif [ $ARG = "--cuda-gdb" ]; then
    CUDAGDB=true
    continue
  elif [ $ARG = "--gdb" ]; then
    GDB=true
    continue
  fi
  set -- "$@" "$ARG"
done

APP_NAME=$(basename {APP_YAML_FILE} .yaml)
if [ $PROFILE ]; then
  if [ $(which nsys) ]; then
    if [ $EXPORT ]; then
      OUTPUT_PATH="/tmp/$APP_NAME"
    else
      OUTPUT_PATH="${PWD}/$APP_NAME"
    fi
    # --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true
    # These config options will be useful to trace unified memory activity but
    # they are disabled by default since they can slow down application performance
    EXECUTABLE_PATH="sudo nsys profile --force-overwrite=true --sample=process-tree --backtrace=dwarf
                    --samples-per-backtrace=2 --cudabacktrace=all --stats=true --output=$OUTPUT_PATH $EXECUTABLE_PATH"
  else
    echo "Could not find nsys"
    exit 1
  fi
elif [ $CUDAGDB ]; then
  if [ $(which cuda-gdb) ]; then
    EXECUTABLE_PATH="cuda-gdb --args $EXECUTABLE_PATH"
  else
    echo "Could not find cuda-gdb"
    exit 1
  fi
elif [ $GDB ]; then
  if [ $(which gdb) ]; then
    EXECUTABLE_PATH="gdb --args $EXECUTABLE_PATH"
  else
    echo "Could not find gdb"
    exit 1
  fi
fi

declare -a graphs={GRAPHS_INPUT}
declare -a sub_graphs={SUB_GRAPHS_INPUT}
# reformat input graphs string
graphs_input=""
for dep in "${graphs[@]}"; do
  graphs_input+="$dep "
done
for dep in "${sub_graphs[@]}"; do
  graphs_input+="$dep "
done

echo "Installing graphs $graphs_input ..."
MANIFEST_PATH=$APP_FOLDER/{MANIFEST_FILE}
$REGISTRY_PATH graph install --graph-files ${graphs_input} --manifest-file-path $MANIFEST_PATH \
                             --target-file-path {TARGET_FILE} --output-directory $APP_FOLDER

# reformat input graphs string
graphs_input=""
for dep in "${graphs[@]}"; do
  graphs_input+="$dep,"
done

echo "Loading graphs $graphs_input ..."
$EXECUTABLE_PATH --app ${graphs_input} --manifest $MANIFEST_PATH $@