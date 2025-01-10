# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

gxf_copy_to_output_dir(
    TARGET_NAME "copy_core_test_apps_data_files"
    FILES
        test_app_GxfGraphLoadFile_component_not_seq.yaml
        test_app_GxfGraphLoadFile_invalid_component_ref.yaml
        test_app_GxfGraphLoadFile_invalid_component_type.yaml
        test_app_GxfGraphLoadFile_invalid_entity_ref.yaml
        test_app_GxfGraphLoadFile_invalid_format.yaml
        test_app_GxfGraphLoadFile_invalid_parameter_value.yaml
        test_app_GxfGraphLoadFile_parameter_not_a_map.yaml
        test_app_GxfGraphLoadFile_unregistered_params.yaml
        test_app_GxfGraphLoadFile_valid.yaml
        test_app_GxfGraphRun_step_count_mismatch.yaml
        test_app_GxfGraphWait.yaml
    ALL
)
