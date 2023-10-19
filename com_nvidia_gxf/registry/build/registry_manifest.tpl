#########################################################################
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
##########################################################################

##############################################
##### GXF Extension Manifest
##############################################

name : {NAME}
extension_library : {EXTENSION_LIBRARY}
uuid : {UUID}
version : {VERSION}
license_file : {LICENSE_FILE}
url : {URL}
git_repository : {GIT_REPOSITORY}
labels : {LABELS}
priority : {PRIORITY}
platform :
  arch : {ARCH}
  os : {OS}
  distribution : {DISTRIBUTION}
compute:
  cuda : {CUDA}
  tensorrt: {TENSORRT}
  cudnn: {CUDNN}
  deepstream: {DEEPSTREAM}
  triton: {TRITON}
  vpi: {VPI}
dependencies : {DEPENDENCIES}
headers : {HEADERS}
binaries : {BINARIES}
python_alias : {PYTHON_ALIAS}
namespace : {NAMESPACE}
python_bindings : {PYTHON_BINDINGS}
python_sources : {PYTHON_SOURCES}
data : {DATA}
