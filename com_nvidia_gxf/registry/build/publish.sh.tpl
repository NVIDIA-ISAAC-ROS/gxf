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

# echo "Script executed from: ${PWD}"

# Find registry cli tool
REGISTRY=registry_cli
REGISTRY_PATH="$(find . -name $REGISTRY -follow)"
if [ ! -f $REGISTRY_PATH ]; then
  echo "Could not find $REGISTRY_PATH"
  exit 1
fi

FORCE={FORCE}

ARGS=""
CUDA={CUDA}
CUDNN={CUDNN}
DEEPSTREAM={DEEPSTREAM}
TRITON={TRITON}
TENSORRT={TENSORRT}
VPI={VPI}

if [ -n "$CUDA" ]; then
    ARGS+=" --cuda {CUDA} "
fi
if [ -n "$CUDNN" ]; then
    ARGS+=" --cudnn {CUDNN} "
fi
if [ -n "$DEEPSTREAM" ]; then
    ARGS+=" --deepstream {DEEPSTREAM} "
fi
if [ -n "$TRITON" ]; then
    ARGS+=" --triton {TRITON} "
fi
if [ -n "$TENSORRT" ]; then
    ARGS+=" --tensorrt {TENSORRT} "
fi
if [ -n "$VPI" ]; then
    ARGS+=" --vpi {VPI} "
fi

echo "Publishing extension {EXTENSION_NAME} ..."
if [ -n "$FORCE" ]; then
    $REGISTRY_PATH extn publish interface --extn-name {EXTENSION_NAME} --repo-name {REPO_NAME} --force
else
    $REGISTRY_PATH extn publish interface --extn-name {EXTENSION_NAME} --repo-name {REPO_NAME}
fi

$REGISTRY_PATH extn publish variant --extn-name {EXTENSION_NAME} --repo-name {REPO_NAME} --arch {ARCH} --os {OS} --distro {DISTRO} ${ARGS}
