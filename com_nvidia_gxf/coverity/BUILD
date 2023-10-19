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
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")

# Configuration setting to enable coverity
config_setting(
    name = "coverity_build",
    define_values = {"run_coverity": "True"},
    visibility = ["//visibility:public"],
)

string_flag(name = "coverity_config", build_setting_default = "autosar-only")

# Configuration setting to configure coverity
config_setting(
    name = "none",
    flag_values = {":coverity_config": "none"},
    visibility = ["//visibility:public"],
)


# Configuration setting to configure coverity
config_setting(
    name = "autosar-only",
    flag_values = {":coverity_config": "autosar-only"},
    visibility = ["//visibility:public"],
)
