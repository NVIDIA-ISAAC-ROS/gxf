#!/bin/sh
#####################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
echo "Script executed from: ${PWD}"
echo "App yaml file is: ${APP_YAML_FILE}"
echo "App manifest file is: ${MANIFEST_YAML_FILE}"

EXECUTABLE=gxe
EXECUTABLE_PATH="$(find . -name $EXECUTABLE -type d -follow)/$EXECUTABLE"

if [ ! -f $EXECUTABLE_PATH ]; then
  echo "Could not find $EXECUTABLE"
  exit 1
fi

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
  elif [ $ARG = "--compute-sanitizer" ]; then
    COMPUTE_SANITIZER=true
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
    EXECUTABLE_PATH="nsys profile --force-overwrite=true --sample=process-tree --backtrace=dwarf
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
elif [ $COMPUTE_SANITIZER ]; then
  if [ $(which compute-sanitizer) ]; then
    EXECUTABLE_PATH="compute-sanitizer $EXECUTABLE_PATH"
  else
    echo "Could not find compute-sanitizer"
    exit 1
  fi
fi

LIB_GXF_CORE_PATH="$(find . -name "libgxf_core.so" -follow)"
for LIB_GXF_CORE_PATH_I in $LIB_GXF_CORE_PATH; do
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"{LIB_PATH}":"$(dirname $LIB_GXF_CORE_PATH_I)"
done

export LD_LIBRARY_PATH

$EXECUTABLE_PATH --app {APP_YAML_FILE} --manifest {MANIFEST_YAML_FILE} $@
