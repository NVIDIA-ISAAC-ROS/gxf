"""
 SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

platform_config_path_selector = {
    "//conditions:default": "gxf_x86_64_cuda_11_8",
    "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_11_8": "gxf_x86_64_cuda_11_8",
    "@com_nvidia_gxf//engine/build:platform_x86_64_cuda_12_1": "gxf_x86_64_cuda_12_1",
    "@com_nvidia_gxf//engine/build:platform_jetpack51": "gxf_jetpack51",
    "@com_nvidia_gxf//engine/build:platform_hp11_sbsa": "gxf_hp11_sbsa",
    "@com_nvidia_gxf//engine/build:platform_hp20_sbsa": "gxf_hp20_sbsa",
    "@com_nvidia_gxf//engine/build:platform_hp21ea_sbsa": "gxf_hp21ea_sbsa",
}
