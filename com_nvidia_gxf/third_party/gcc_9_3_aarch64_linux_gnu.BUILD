"""
 SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

"""

# GNU GCC 9.3.0 cross aarch64 compiler

filegroup(
    name = "gcc_9_3_aarch64_linux_gnu",
    srcs = glob([
        "bin/*",
        "aarch64-buildroot-linux-gnu/include/c++/9.3.0/**/*",
        "aarch64-buildroot-linux-gnu/sysroot/usr/include/**/*",
        "lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/include/**/*",
        "lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/include-fixed/**/*",
        "aarch64-buildroot-linux-gnu/sysroot/usr/lib/*",
    ]) + [
        "aarch64-buildroot-linux-gnu/sysroot/lib64",
        "aarch64-buildroot-linux-gnu/sysroot/usr/lib64",
    ],
    visibility = ["//visibility:public"],
)
