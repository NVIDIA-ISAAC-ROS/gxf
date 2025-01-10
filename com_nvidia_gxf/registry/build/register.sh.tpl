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
REGISTRY_PATH="$(find ${PWD} -name $REGISTRY -follow)"
if [ ! -f $REGISTRY_PATH ]; then
  echo "Could not find $REGISTRY_PATH"
  exit 1
fi

declare -a dependencies={DEPENDENCIES}
for dep in "${dependencies[@]}"; do
  echo "Registering dependency: $dep"
  $REGISTRY_PATH extn add --manifest-name $dep
done

echo "Loading manifest {EXTENSION_MANIFEST} ..."
$REGISTRY_PATH extn add --manifest-name {EXTENSION_MANIFEST} --metadata-file {EXTENSION_METADATA}
