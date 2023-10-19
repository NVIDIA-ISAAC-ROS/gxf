
#!/bin/bash
#####################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
set +e

WORK_DIR=/tmp/minidump

if [[ $1 == "--use_dazel" ]]; then
    BAZEL=dazel
else
    BAZEL=bazel
fi

# Creates working folder, builds tools from breakpad and copies them to working folder
if [ ! -d "$WORK_DIR" ]; then
    mkdir -p "$WORK_DIR"
fi

# Creates binaries of tools if not present
if [ ! -f "${WORK_DIR}/minidump_stackwalk"  -o  ! -f "${WORK_DIR}/dump_syms" ]; then
    $BAZEL build @breakpad//:minidump_stackwalk @breakpad//:dump_syms
    rm -f "${WORK_DIR}/minidump_stackwalk" "${WORK_DIR}/dump_syms"
    cp bazel-bin/external/breakpad/dump_syms bazel-bin/external/breakpad/minidump_stackwalk $WORK_DIR
fi
